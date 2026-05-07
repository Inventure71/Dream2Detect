create extension if not exists pgcrypto;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'dream2detect-images',
  'dream2detect-images',
  true,
  20971520,
  array['image/jpeg', 'image/png', 'image/webp']
)
on conflict (id) do update
set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "public can read dream2detect images" on storage.objects;
create policy "public can read dream2detect images"
  on storage.objects for select
  to public
  using (bucket_id = 'dream2detect-images');

create table if not exists public.labeling_datasets (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  name text not null,
  kind text not null check (kind in ('real', 'synthetic')),
  source_url text,
  storage_bucket text not null default 'dream2detect-images',
  storage_prefix text not null,
  status text not null default 'active' check (status in ('active', 'archived')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.labeling_images (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid not null references public.labeling_datasets(id) on delete cascade,
  image_id text not null,
  source_image_id text not null,
  storage_bucket text not null default 'dream2detect-images',
  storage_path text not null,
  local_image_path text,
  source_dataset text,
  original_label text,
  prompt_id text,
  prompt_text text,
  intended_score_band text,
  intended_coarse_class text,
  intended_representative_score integer,
  status text not null default 'pending'
    check (status in ('pending', 'in_review', 'labeled', 'needs_second_review', 'rejected')),
  lock_owner uuid references auth.users(id) on delete set null,
  lock_reviewer_name text,
  lock_previous_status text check (lock_previous_status in ('pending', 'needs_second_review')),
  lock_expires_at timestamptz,
  version integer not null default 1,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (dataset_id, image_id)
);

create index if not exists labeling_images_dataset_status_idx
  on public.labeling_images(dataset_id, status, source_image_id);

create index if not exists labeling_images_lock_idx
  on public.labeling_images(lock_owner, lock_expires_at);

create index if not exists labeling_images_available_idx
  on public.labeling_images(dataset_id, source_image_id)
  where status in ('pending', 'needs_second_review');

create table if not exists public.label_reviews (
  id uuid primary key default gen_random_uuid(),
  image_row_id uuid not null references public.labeling_images(id) on delete cascade,
  reviewer_id uuid references auth.users(id) on delete set null,
  reviewer_name text,
  review_type text not null default 'human'
    check (review_type in ('human', 'llm_assist', 'synthetic_qc')),
  score_band text,
  coarse_class text,
  representative_score integer,
  decision text not null
    check (decision in ('approved', 'corrected', 'needs_second_review', 'rejected', 'skipped')),
  uncertainty_flag boolean not null default false,
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists label_reviews_image_row_idx
  on public.label_reviews(image_row_id, created_at desc);

create table if not exists public.final_labels (
  image_row_id uuid primary key references public.labeling_images(id) on delete cascade,
  final_score_band text,
  final_coarse_class text,
  final_representative_score integer,
  decision text not null
    check (decision in ('approved', 'corrected', 'needs_second_review', 'rejected')),
  uncertainty_flag boolean not null default false,
  notes text,
  finalized_by uuid references auth.users(id) on delete set null,
  finalized_by_name text,
  finalized_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.coarse_class_for_band(score_band text)
returns text
language sql
immutable
as $$
  select case score_band
    when '0-10' then 'intact'
    when '11-20' then 'minor'
    when '21-30' then 'minor'
    when '31-35' then 'minor'
    when '36-45' then 'moderate'
    when '46-55' then 'moderate'
    when '56-65' then 'moderate'
    when '66-75' then 'severe'
    when '76-85' then 'severe'
    when '86-100' then 'severe'
    else null
  end;
$$;

create or replace function public.representative_score_for_band(score_band text)
returns integer
language sql
immutable
as $$
  select case score_band
    when '0-10' then 5
    when '11-20' then 15
    when '21-30' then 25
    when '31-35' then 33
    when '36-45' then 40
    when '46-55' then 50
    when '56-65' then 60
    when '66-75' then 70
    when '76-85' then 80
    when '86-100' then 93
    else null
  end;
$$;

create or replace function public.claim_next_image(
  p_dataset_slug text,
  p_reviewer_name text default null
)
returns public.labeling_images
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user uuid := auth.uid();
  v_image public.labeling_images%rowtype;
begin
  if v_user is null then
    raise exception 'Authentication required';
  end if;

  update public.labeling_images li
  set
    status = 'in_review',
    lock_owner = v_user,
    lock_reviewer_name = p_reviewer_name,
    lock_previous_status = case
      when li.status in ('pending', 'needs_second_review') then li.status
      else coalesce(li.lock_previous_status, 'pending')
    end,
    lock_expires_at = now() + interval '30 minutes',
    version = li.version + 1,
    updated_at = now()
  where li.id = (
    select li2.id
    from public.labeling_images li2
    join public.labeling_datasets d on d.id = li2.dataset_id
    where d.slug = p_dataset_slug
      and d.status = 'active'
      and (
        li2.status in ('pending', 'needs_second_review')
        or (li2.status = 'in_review' and li2.lock_expires_at < now())
      )
      and (
        li2.lock_owner is null
        or li2.lock_owner = v_user
        or li2.lock_expires_at < now()
      )
    order by
      case when li2.status = 'needs_second_review' then 0 else 1 end,
      li2.source_image_id
    limit 1
    for update skip locked
  )
  returning * into v_image;

  return v_image;
end;
$$;

create or replace function public.release_image_lock(p_image_row_id uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user uuid := auth.uid();
begin
  if v_user is null then
    raise exception 'Authentication required';
  end if;

  update public.labeling_images
  set
    status = case
      when status = 'in_review' then coalesce(lock_previous_status, 'pending')
      else status
    end,
    lock_owner = null,
    lock_reviewer_name = null,
    lock_previous_status = null,
    lock_expires_at = null,
    version = version + 1,
    updated_at = now()
  where id = p_image_row_id
    and lock_owner = v_user;
end;
$$;

create or replace function public.save_label_review(
  p_image_row_id uuid,
  p_version integer,
  p_reviewer_name text,
  p_score_band text,
  p_decision text,
  p_uncertainty_flag boolean,
  p_notes text
)
returns public.labeling_images
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user uuid := auth.uid();
  v_image public.labeling_images%rowtype;
  v_coarse text := public.coarse_class_for_band(p_score_band);
  v_rep integer := public.representative_score_for_band(p_score_band);
  v_next_status text;
begin
  if v_user is null then
    raise exception 'Authentication required';
  end if;

  select *
  into v_image
  from public.labeling_images
  where id = p_image_row_id
  for update;

  if not found then
    raise exception 'Image not found';
  end if;

  if v_image.version <> p_version then
    raise exception 'Stale image version';
  end if;

  if v_image.lock_owner is not null and v_image.lock_owner <> v_user and v_image.lock_expires_at > now() then
    raise exception 'Image is locked by another reviewer';
  end if;

  if p_decision <> 'rejected' and p_score_band is null then
    raise exception 'score_band is required unless rejecting the image';
  end if;

  if p_decision <> 'rejected' and (v_coarse is null or v_rep is null) then
    raise exception 'Unknown score_band: %', p_score_band;
  end if;

  insert into public.label_reviews (
    image_row_id,
    reviewer_id,
    reviewer_name,
    score_band,
    coarse_class,
    representative_score,
    decision,
    uncertainty_flag,
    notes
  )
  values (
    p_image_row_id,
    v_user,
    p_reviewer_name,
    p_score_band,
    v_coarse,
    v_rep,
    p_decision,
    coalesce(p_uncertainty_flag, false),
    p_notes
  );

  v_next_status := case
    when p_decision = 'rejected' then 'rejected'
    when p_decision = 'needs_second_review' or coalesce(p_uncertainty_flag, false) then 'needs_second_review'
    else 'labeled'
  end;

  insert into public.final_labels (
    image_row_id,
    final_score_band,
    final_coarse_class,
    final_representative_score,
    decision,
    uncertainty_flag,
    notes,
    finalized_by,
    finalized_by_name
  )
  values (
    p_image_row_id,
    p_score_band,
    v_coarse,
    v_rep,
    case when p_decision = 'skipped' then 'needs_second_review' else p_decision end,
    coalesce(p_uncertainty_flag, false),
    p_notes,
    v_user,
    p_reviewer_name
  )
  on conflict (image_row_id) do update
  set
    final_score_band = excluded.final_score_band,
    final_coarse_class = excluded.final_coarse_class,
    final_representative_score = excluded.final_representative_score,
    decision = excluded.decision,
    uncertainty_flag = excluded.uncertainty_flag,
    notes = excluded.notes,
    finalized_by = excluded.finalized_by,
    finalized_by_name = excluded.finalized_by_name,
    updated_at = now();

  update public.labeling_images
  set
    status = v_next_status,
    lock_owner = null,
    lock_reviewer_name = null,
    lock_previous_status = null,
    lock_expires_at = null,
    version = version + 1,
    updated_at = now()
  where id = p_image_row_id
  returning * into v_image;

  return v_image;
end;
$$;

create or replace view public.dataset_progress as
select
  d.slug,
  d.name,
  d.kind,
  count(i.id)::integer as total,
  count(*) filter (where i.status = 'pending')::integer as pending,
  count(*) filter (where i.status = 'in_review')::integer as in_review,
  count(*) filter (where i.status = 'labeled')::integer as labeled,
  count(*) filter (where i.status = 'needs_second_review')::integer as needs_second_review,
  count(*) filter (where i.status = 'rejected')::integer as rejected
from public.labeling_datasets d
left join public.labeling_images i on i.dataset_id = d.id
group by d.id, d.slug, d.name, d.kind;

create or replace view public.real_relabel_export as
select
  i.image_id,
  i.local_image_path as image_path,
  i.source_dataset,
  i.original_label,
  ''::text as llm_score_band,
  null::integer as llm_representative_score,
  ''::text as llm_coarse_class,
  ''::text as llm_reasoning,
  fl.final_score_band as human_score_band,
  fl.final_representative_score as human_representative_score,
  fl.final_coarse_class as human_coarse_class,
  fl.final_score_band,
  fl.final_representative_score,
  fl.final_coarse_class,
  i.status as review_status,
  case when fl.uncertainty_flag then 'true' else '' end as disagreement_flag,
  fl.notes as review_notes,
  ''::text as split
from public.labeling_images i
join public.labeling_datasets d on d.id = i.dataset_id
left join public.final_labels fl on fl.image_row_id = i.id
where d.kind = 'real';

alter table public.labeling_datasets enable row level security;
alter table public.labeling_images enable row level security;
alter table public.label_reviews enable row level security;
alter table public.final_labels enable row level security;

drop policy if exists "authenticated users can read datasets" on public.labeling_datasets;
create policy "authenticated users can read datasets"
  on public.labeling_datasets for select
  to authenticated
  using (true);

drop policy if exists "authenticated users can read images" on public.labeling_images;
create policy "authenticated users can read images"
  on public.labeling_images for select
  to authenticated
  using (true);

drop policy if exists "authenticated users can read reviews" on public.label_reviews;
create policy "authenticated users can read reviews"
  on public.label_reviews for select
  to authenticated
  using (true);

drop policy if exists "authenticated users can read final labels" on public.final_labels;
create policy "authenticated users can read final labels"
  on public.final_labels for select
  to authenticated
  using (true);

grant usage on schema public to authenticated;
grant select on public.labeling_datasets to authenticated;
grant select on public.labeling_images to authenticated;
grant select on public.label_reviews to authenticated;
grant select on public.final_labels to authenticated;
grant select on public.dataset_progress to authenticated;
grant execute on function public.claim_next_image(text, text) to authenticated;
grant execute on function public.release_image_lock(uuid) to authenticated;
grant execute on function public.save_label_review(uuid, integer, text, text, text, boolean, text) to authenticated;
