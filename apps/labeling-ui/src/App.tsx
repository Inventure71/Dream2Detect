import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  Clock3,
  Database,
  LogOut,
  RefreshCw,
  Search,
  SkipForward,
  X,
} from "lucide-react";
import { BAND_REFERENCES, deriveFromBand, type ScoreBand } from "./lib/bands";
import { defaultDatasetSlug, isSupabaseConfigured, supabase } from "./lib/supabase";
import type { DatasetProgress, LabelingImage, SaveDecision } from "./types";

type SessionUser = {
  id: string;
  email?: string;
};

type Notice = {
  tone: "info" | "error" | "success";
  text: string;
};

function getStorageUrl(image: LabelingImage) {
  if (!supabase) return "";
  const { data } = supabase.storage
    .from(image.storage_bucket)
    .getPublicUrl(image.storage_path);
  return data.publicUrl;
}

function reviewerName(user: SessionUser | null) {
  return user?.email ?? "reviewer";
}

export default function App() {
  const [user, setUser] = useState<SessionUser | null>(null);
  const [email, setEmail] = useState("");
  const [datasetSlug, setDatasetSlug] = useState(defaultDatasetSlug);
  const [progress, setProgress] = useState<DatasetProgress | null>(null);
  const [image, setImage] = useState<LabelingImage | null>(null);
  const [history, setHistory] = useState<LabelingImage[]>([]);
  const [selectedBand, setSelectedBand] = useState<ScoreBand | null>(null);
  const [uncertain, setUncertain] = useState(false);
  const [notes, setNotes] = useState("");
  const [notice, setNotice] = useState<Notice | null>(null);
  const [loading, setLoading] = useState(false);

  const derived = useMemo(() => {
    if (!selectedBand) return null;
    return deriveFromBand(selectedBand);
  }, [selectedBand]);

  const imageUrl = image ? getStorageUrl(image) : null;

  useEffect(() => {
    if (!supabase) return;
    const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const authError = hashParams.get("error_description") || hashParams.get("error");
    if (authError) {
      setNotice({
        tone: "error",
        text: authError.replaceAll("+", " "),
      });
      window.history.replaceState(null, "", window.location.pathname);
    }

    supabase.auth.getUser().then(({ data }) => {
      if (data.user) {
        setUser({ id: data.user.id, email: data.user.email });
      }
    });

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(
        session?.user
          ? { id: session.user.id, email: session.user.email }
          : null,
      );
    });

    return () => authListener.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (!user) return;
    void loadProgress();
    void loadCurrentLock();
  }, [user, datasetSlug]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      const keyMap: Record<string, ScoreBand> = {
        "1": "0-10",
        "2": "11-20",
        "3": "21-30",
        "4": "31-35",
        "5": "36-45",
        "6": "46-55",
        "7": "56-65",
        "8": "66-75",
        "9": "76-85",
        "0": "86-100",
      };

      if (keyMap[event.key]) {
        setSelectedBand(keyMap[event.key]);
      }

      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        void saveReview("approved", true);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [image, selectedBand, uncertain, notes]);

  async function signIn() {
    if (!supabase) return;
    setLoading(true);
    setNotice(null);
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: window.location.origin,
      },
    });
    setLoading(false);
    if (error) {
      setNotice({ tone: "error", text: error.message });
      return;
    }
    setNotice({ tone: "success", text: "Check your email for the sign-in link." });
  }

  async function signOut() {
    if (!supabase) return;
    await supabase.auth.signOut();
    setUser(null);
    setImage(null);
    setProgress(null);
  }

  async function loadProgress() {
    if (!supabase) return;
    const { data, error } = await supabase
      .from("dataset_progress")
      .select("*")
      .eq("slug", datasetSlug)
      .maybeSingle();

    if (error) {
      setNotice({ tone: "error", text: error.message });
      return;
    }

    setProgress((data as DatasetProgress | null) ?? null);
  }

  async function loadCurrentLock() {
    if (!user || !supabase) return;

    const { data, error } = await supabase
      .from("labeling_images")
      .select("*, labeling_datasets!inner(slug)")
      .eq("labeling_datasets.slug", datasetSlug)
      .eq("lock_owner", user.id)
      .gt("lock_expires_at", new Date().toISOString())
      .limit(1)
      .maybeSingle();

    if (error) {
      setNotice({ tone: "error", text: error.message });
      return;
    }

    if (data) {
      startImage(data as LabelingImage);
    }
  }

  function startImage(nextImage: LabelingImage) {
    setImage((current) => {
      if (current && current.id !== nextImage.id) {
        setHistory((items) => [current, ...items.filter((item) => item.id !== current.id)].slice(0, 20));
      }
      return nextImage;
    });
    setSelectedBand((nextImage.intended_score_band as ScoreBand | null) ?? null);
    setUncertain(false);
    setNotes("");
    setNotice(null);
  }

  async function claimNext(options: { releaseCurrentFirst?: boolean } = {}) {
    if (!supabase) return;
    const currentImage = image;
    setLoading(true);
    setNotice(null);

    if (options.releaseCurrentFirst && currentImage) {
      const { error: releaseError } = await supabase.rpc("release_image_lock", {
        p_image_row_id: currentImage.id,
      });

      if (releaseError) {
        setLoading(false);
        setNotice({ tone: "error", text: releaseError.message });
        return;
      }

      setImage(null);
      setSelectedBand(null);
      setUncertain(false);
      setNotes("");
    }

    const { data, error } = await supabase.rpc("claim_next_image", {
      p_dataset_slug: datasetSlug,
      p_reviewer_name: reviewerName(user),
    });
    setLoading(false);

    if (error) {
      setNotice({ tone: "error", text: error.message });
      return;
    }

    if (!data) {
      setNotice({ tone: "info", text: "No available images in this dataset." });
      await loadProgress();
      return;
    }

    startImage(data as LabelingImage);
    await loadProgress();
  }

  async function releaseCurrent() {
    if (!image || !supabase) return;
    setLoading(true);
    const { error } = await supabase.rpc("release_image_lock", {
      p_image_row_id: image.id,
    });
    setLoading(false);

    if (error) {
      setNotice({ tone: "error", text: error.message });
      return;
    }

    setImage(null);
    setSelectedBand(null);
    await loadProgress();
  }

  async function saveReview(decision: SaveDecision, moveNext: boolean) {
    if (!supabase) return;
    if (!image) return;
    if (decision !== "rejected" && !selectedBand) {
      setNotice({ tone: "error", text: "Select a score band first." });
      return;
    }

    setLoading(true);
    setNotice(null);
    const { error } = await supabase.rpc("save_label_review", {
      p_image_row_id: image.id,
      p_version: image.version,
      p_reviewer_name: reviewerName(user),
      p_score_band: decision === "rejected" ? null : selectedBand,
      p_decision: decision,
      p_uncertainty_flag: uncertain,
      p_notes: notes.trim() || null,
    });
    setLoading(false);

    if (error) {
      setNotice({ tone: "error", text: error.message });
      return;
    }

    setNotice({ tone: "success", text: "Saved." });
    setImage(null);
    setSelectedBand(null);
    setUncertain(false);
    setNotes("");
    await loadProgress();

    if (moveNext) {
      await claimNext();
    }
  }

  function restorePrevious() {
    const [previous, ...rest] = history;
    if (!previous) return;
    setHistory(rest);
    startImage(previous);
  }

  if (!isSupabaseConfigured) {
    return (
      <main className="loginShell">
        <section className="loginPanel">
          <div className="brandRow">
            <Database size={28} />
            <div>
              <h1>Dream2Detect Labeling</h1>
              <p>Supabase configuration missing</p>
            </div>
          </div>
          <p className="notice info">
            Create <code>apps/labeling-ui/.env</code> from <code>.env.example</code>.
          </p>
        </section>
      </main>
    );
  }

  if (!user) {
    return (
      <main className="loginShell">
        <section className="loginPanel">
          <div className="brandRow">
            <Database size={28} />
            <div>
              <h1>Dream2Detect Labeling</h1>
              <p>Shared severity review workspace</p>
            </div>
          </div>
          <label className="fieldLabel" htmlFor="email">
            Email
          </label>
          <div className="loginForm">
            <input
              id="email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="teammate@example.com"
            />
            <button onClick={signIn} disabled={!email || loading}>
              <Check size={18} />
              Sign in
            </button>
          </div>
          {notice && <p className={`notice ${notice.tone}`}>{notice.text}</p>}
        </section>
      </main>
    );
  }

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="topIdentity">
          <Database size={22} />
          <div>
            <h1>Dream2Detect Labeling</h1>
            <span>{user.email}</span>
          </div>
        </div>
        <div className="topControls">
          <div className="datasetControl">
            <Search size={16} />
            <input
              value={datasetSlug}
              onChange={(event) => setDatasetSlug(event.target.value)}
              aria-label="Dataset slug"
            />
          </div>
          <button className="iconButton" onClick={loadProgress} aria-label="Refresh progress">
            <RefreshCw size={18} />
          </button>
          <button className="iconButton" onClick={signOut} aria-label="Sign out">
            <LogOut size={18} />
          </button>
        </div>
      </header>

      <section className="progressBand">
        {progress ? (
          <>
            <ProgressItem label="Total" value={progress.total} />
            <ProgressItem label="Pending" value={progress.pending} />
            <ProgressItem label="In Review" value={progress.in_review} />
            <ProgressItem label="Labeled" value={progress.labeled} />
            <ProgressItem label="Second Review" value={progress.needs_second_review} />
            <ProgressItem label="Rejected" value={progress.rejected} />
          </>
        ) : (
          <span className="emptyState">No dataset progress loaded.</span>
        )}
      </section>

      {notice && <p className={`notice ${notice.tone}`}>{notice.text}</p>}

      <section className="exampleStrip" aria-label="Band image examples">
        {BAND_REFERENCES.map((item) => (
          <button
            key={item.band}
            className={`exampleTile ${selectedBand === item.band ? "active" : ""}`}
            onClick={() => setSelectedBand(item.band)}
            title={`${item.band} ${item.coarse}`}
          >
            <img src={item.exampleImage} alt={`${item.band} ${item.coarse} example`} />
            <span>
              <strong>{item.band}</strong>
              <small>{item.exampleSource === "reviewed" ? item.coarse : "unreviewed guide"}</small>
            </span>
          </button>
        ))}
      </section>

      <section className="workspace">
        <aside className="referencePanel">
          <h2>Band Examples</h2>
          <div className="bandList">
            {BAND_REFERENCES.map((item, index) => (
              <button
                key={item.band}
                className={`bandReference ${selectedBand === item.band ? "active" : ""}`}
                onClick={() => setSelectedBand(item.band)}
              >
                <span className="bandShortcut">{index === 9 ? 0 : index + 1}</span>
                <span>
                  <strong>{item.band}</strong>
                  <small>{item.coarse} · {item.representativeScore}</small>
                  <em>{item.cue}</em>
                </span>
              </button>
            ))}
          </div>
        </aside>

        <section className="imageStage">
          {image && imageUrl ? (
            <>
              <div className="imageMetaBar">
                <div>
                  <span className="eyebrow">{image.status.replaceAll("_", " ")}</span>
                  <h2>{image.image_id}</h2>
                </div>
                <span className="lockBadge">
                  <Clock3 size={15} />
                  {image.lock_expires_at ? new Date(image.lock_expires_at).toLocaleTimeString() : "locked"}
                </span>
              </div>
              <div className="imageFrame">
                <img src={imageUrl} alt={image.image_id} />
              </div>
            </>
          ) : (
            <div className="startPanel">
              <Database size={40} />
              <h2>{progress?.name ?? datasetSlug}</h2>
              <button onClick={() => claimNext()} disabled={loading}>
                <ArrowRight size={18} />
                Get next image
              </button>
            </div>
          )}
        </section>

        <aside className="reviewPanel">
          <section className="hintBlock">
            <h2>Known Hints</h2>
            <Hint label="Source" value={image?.source_dataset} />
            <Hint label="Original" value={image?.original_label} />
            <Hint label="Intended Band" value={image?.intended_score_band} />
            <Hint label="Intended Class" value={image?.intended_coarse_class} />
            <Hint label="Prompt" value={image?.prompt_text} />
          </section>

          <section className="selectionBlock">
            <h2>Selected Label</h2>
            {selectedBand && derived ? (
              <div className="selectedSummary">
                <strong>{selectedBand}</strong>
                <span>{derived.coarseClass}</span>
                <span>{derived.representativeScore}</span>
              </div>
            ) : (
              <span className="emptyState">No band selected.</span>
            )}
            <label className="checkRow">
              <input
                type="checkbox"
                checked={uncertain}
                onChange={(event) => setUncertain(event.target.checked)}
              />
              Boundary or uncertain
            </label>
            <textarea
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              placeholder="Review notes"
            />
          </section>

          <section className="actionGrid">
            <button onClick={() => saveReview("approved", true)} disabled={!image || loading}>
              <Check size={18} />
              Save Next
            </button>
            <button onClick={() => saveReview("needs_second_review", true)} disabled={!image || loading}>
              <AlertTriangle size={18} />
              Second Review
            </button>
            <button onClick={() => saveReview("rejected", true)} disabled={!image || loading}>
              <X size={18} />
              Reject
            </button>
            <button onClick={releaseCurrent} disabled={!image || loading}>
              <SkipForward size={18} />
              Skip
            </button>
          </section>

          <section className="navRow">
            <button onClick={restorePrevious} disabled={!history.length || loading}>
              <ArrowLeft size={18} />
              Previous
            </button>
            <button onClick={() => claimNext({ releaseCurrentFirst: true })} disabled={loading}>
              <ArrowRight size={18} />
              Next
            </button>
          </section>
        </aside>
      </section>
    </main>
  );
}

function ProgressItem({ label, value }: { label: string; value: number }) {
  return (
    <div className="progressItem">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Hint({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) return null;
  return (
    <div className="hint">
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}
