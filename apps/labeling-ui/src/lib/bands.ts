export type ScoreBand =
  | "0-10"
  | "11-20"
  | "21-30"
  | "31-35"
  | "36-45"
  | "46-55"
  | "56-65"
  | "66-75"
  | "76-85"
  | "86-100";

export type CoarseClass = "intact" | "minor" | "moderate" | "severe";

export type BandReference = {
  band: ScoreBand;
  coarse: CoarseClass;
  representativeScore: number;
  cue: string;
  exampleImage: string;
  exampleSource: "reviewed" | "intended_unreviewed";
};

export const BAND_REFERENCES: BandReference[] = [
  {
    band: "0-10",
    coarse: "intact",
    representativeScore: 5,
    cue: "Visually normal package; only negligible marks.",
    exampleImage: "/band-examples/0_10.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "11-20",
    coarse: "minor",
    representativeScore: 15,
    cue: "Light visible defect; small dent, tiny tear, or slight corner wear.",
    exampleImage: "/band-examples/11_20.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "21-30",
    coarse: "minor",
    representativeScore: 25,
    cue: "Clear minor damage, still mostly normal in shape.",
    exampleImage: "/band-examples/21_30.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "31-35",
    coarse: "minor",
    representativeScore: 33,
    cue: "High minor case near moderate; obvious but not structurally serious.",
    exampleImage: "/band-examples/31_35.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "36-45",
    coarse: "moderate",
    representativeScore: 40,
    cue: "Meaningful visible damage; noticeable dent, edge damage, or small opening.",
    exampleImage: "/band-examples/36_45.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "46-55",
    coarse: "moderate",
    representativeScore: 50,
    cue: "Multiple or stronger defects; package condition clearly degraded.",
    exampleImage: "/band-examples/46_55.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "56-65",
    coarse: "moderate",
    representativeScore: 60,
    cue: "High moderate case; substantial deformation without full collapse.",
    exampleImage: "/band-examples/56_65.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "66-75",
    coarse: "severe",
    representativeScore: 70,
    cue: "Major visible structural damage; strong crushing or serious tear.",
    exampleImage: "/band-examples/66_75.jpg",
    exampleSource: "reviewed",
  },
  {
    band: "76-85",
    coarse: "severe",
    representativeScore: 80,
    cue: "Very damaged package; large opening, collapse, or major material failure.",
    exampleImage: "/band-examples/76_85.jpg",
    exampleSource: "intended_unreviewed",
  },
  {
    band: "86-100",
    coarse: "severe",
    representativeScore: 93,
    cue: "Extreme visible damage while still recognizable as packaging.",
    exampleImage: "/band-examples/86_100.jpg",
    exampleSource: "reviewed",
  },
];

export function deriveFromBand(scoreBand: ScoreBand) {
  const match = BAND_REFERENCES.find((item) => item.band === scoreBand);
  if (!match) {
    throw new Error(`Unknown score band: ${scoreBand}`);
  }
  return {
    coarseClass: match.coarse,
    representativeScore: match.representativeScore,
  };
}
