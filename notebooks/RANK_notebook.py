# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 2 -- RANK
# **This is the submission notebook -- must run in <=5 minutes, CPU only, no internet.**
#
# Requirements before running:
# - `PRECOMPUTE_notebook.ipynb` must have been run first (with the updated Cell 6 that stores reasoning fields)
# - `artifacts/` folder must contain the 4 saved files
#
# **Key change:** this notebook NEVER reads `candidates.jsonl`. Everything needed
# for scoring AND reasoning strings is pulled from `features.parquet`, which was
# built once during precompute. This removes the slow full-file scan and makes
# runtime independent of where matching candidates fall in the file.
#
# **Expected total runtime: ~5-15 seconds.**

# %% [markdown]
# ## Cell 1 -- Imports

# %%
import csv, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
t_start = time.time()
print('Imports OK')

# %% [markdown]
# ## Cell 2 -- Config

# %%
ARTIFACTS_DIR = '/home/richa-kumari/Documents/redrob/artifacts'   # must match PRECOMPUTE_notebook
OUTPUT_CSV    = '/home/richa-kumari/Documents/redrob/final_submission.csv'
EMBED_WEIGHT  = 0.25            # 25% semantic, 75% structured
TOP_N         = 100

print(f'Artifacts dir exists: {Path(ARTIFACTS_DIR).exists()}')
for fname in ['features.parquet','embeddings.npy','candidate_ids.npy','jd_embedding.npy']:
    exists = Path(f'{ARTIFACTS_DIR}/{fname}').exists()
    print(f'  {"OK" if exists else "MISSING"}  {fname}')


# %% [markdown]
# ## Cell 3 -- Reasoning generator (uses only parquet fields, no jsonl)

# %%
def generate_reasoning(row, comps):
    title   = row.current_title or 'Unknown'
    company = row.current_company or ''
    yoe     = row.yoe or 0
    loc     = row.location or 'Unknown'
    top_skills = row.top_ai_skills or 'no core AI skills'

    rrr  = row.response_rate
    gh   = row.github_score
    np_d = row.notice_days or 0
    otw  = bool(row.open_to_work)

    warns = []
    if comps['honeypot_mult'] < 0.5:     warns.append('profile inconsistencies')
    if comps['availability_mult'] < 0.35: warns.append('low availability')
    if np_d >= 120: warns.append(f'{np_d}-day notice')

    s1 = f"{title} @ {company}, {yoe:.1f}y; AI skills: {top_skills}; {loc}."

    extras = []
    if otw: extras.append('actively looking')
    if rrr is not None and rrr >= 0: extras.append(f'response {rrr:.0%}')
    if gh is not None and gh > 0: extras.append(f'GitHub {gh:.0f}/100')
    if warns: extras.append('Caution: ' + '; '.join(warns))

    s2 = ('; '.join(extras)).capitalize() + '.' if extras else ''
    return (s1 + ' ' + s2).strip()

print('Reasoning generator ready (parquet-only, no file scan)')

# %% [markdown]
# ## Cell 4 -- Load artifacts from disk

# %%
t0 = time.time()
print('Loading artifacts ...')
features_df   = pd.read_parquet(f'{ARTIFACTS_DIR}/features.parquet')
all_embs      = np.load(f'{ARTIFACTS_DIR}/embeddings.npy')
candidate_ids = np.load(f'{ARTIFACTS_DIR}/candidate_ids.npy')
jd_embedding  = np.load(f'{ARTIFACTS_DIR}/jd_embedding.npy')
print(f'Loaded in {time.time()-t0:.2f}s')
print(f'  features_df   : {features_df.shape}')
print(f'  all_embs      : {all_embs.shape}')
print(f'  candidate_ids : {candidate_ids.shape}')
print(f'  jd_embedding  : {jd_embedding.shape}')

# Sanity check: required reasoning columns must be present (from updated precompute)
required_cols = {'current_title','current_company','location','yoe',
                  'response_rate','github_score','notice_days','open_to_work','top_ai_skills'}
missing = required_cols - set(features_df.columns)
if missing:
    raise ValueError(f'features.parquet is missing columns: {missing}. '
                      f'Re-run the updated PRECOMPUTE_notebook.ipynb Cell 6.')
print('All required reasoning columns present ✓')

# %% [markdown]
# ## Cell 5 -- Compute final scores (vectorised, instant)

# %%
t0 = time.time()

# Embeddings are L2-normalised -> dot product = cosine similarity
sims = (all_embs @ jd_embedding).astype(float)
features_df['embedding_sim'] = sims

features_df['final_score'] = (
    features_df['structured'] * (1 - EMBED_WEIGHT) +
    features_df['embedding_sim'] * EMBED_WEIGHT
) * features_df['honeypot_mult'] * features_df['availability_mult']

print(f'Scored {len(features_df):,} candidates in {time.time()-t0:.3f}s')
print(f'Score range: {features_df.final_score.min():.4f} - {features_df.final_score.max():.4f}')

# %% [markdown]
# ## Cell 6 -- Validate top 100

# %%
top100 = features_df.nlargest(TOP_N, 'final_score')
print('=== TOP 100 VALIDATION ===')
print(f"Honeypot=0 in top 100  : {(top100.honeypot_mult==0).sum()}  <- must be 0")
print(f"Score non-increasing   : {(top100.final_score.values[:-1]>=top100.final_score.values[1:]).all()}")
print(f"Unique candidate IDs   : {top100.candidate_id.nunique()}  <- must be 100")
print()
print('Top 10 candidates:')
display(top100.head(10)[['candidate_id','current_title','yoe','final_score',
                          'title_score','career_score','skill_score','embedding_sim']])

# %% [markdown]
# ## Cell 7 -- Build submission rows and save submission.csv (no file I/O on candidates.jsonl)

# %%
t0 = time.time()
top100_sorted = features_df.nlargest(TOP_N, 'final_score').reset_index(drop=True)
submission_rows = []

for rank, row in enumerate(top100_sorted.itertuples(), start=1):
    score = max(0.0, min(1.0, float(row.final_score)))
    comps = {
        'honeypot_mult':     row.honeypot_mult,
        'availability_mult': row.availability_mult,
    }
    reasoning = generate_reasoning(row, comps)
    submission_rows.append({
        'candidate_id': row.candidate_id,
        'rank':         rank,
        'score':        round(score, 4),
        'reasoning':    reasoning,
    })

# Assertions
assert len(submission_rows) == TOP_N
scores = [r['score'] for r in submission_rows]
assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), 'Scores not non-increasing!'
assert len({r['candidate_id'] for r in submission_rows}) == TOP_N, 'Duplicate IDs!'

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['candidate_id','rank','score','reasoning'])
    writer.writeheader()
    writer.writerows(submission_rows)

print(f'Built + saved {OUTPUT_CSV} in {time.time()-t0:.2f}s')
print('All assertions passed')

total_time = time.time() - t_start
print()
print(f'Total wall-clock time: {total_time:.2f}s ({total_time/60:.3f} min)')
if total_time < 300:
    print(f'Within 5-minute budget ({300-total_time:.0f}s to spare)')
else:
    print(f'EXCEEDED 5-minute budget!')

# %% [markdown]
# ## Cell 8 -- Preview results

# %%
sub_df = pd.read_csv(OUTPUT_CSV)
print('submission.csv -- top 10:')
display(sub_df.head(10))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sub_df['score'].plot(ax=axes[0], color='steelblue', marker='o', markersize=3)
axes[0].set_title('Score by Rank (should be decreasing)')
axes[0].set_xlabel('Rank'); axes[0].set_ylabel('Score')
top100_sorted['current_title'].value_counts().head(10).plot(kind='barh', ax=axes[1], color='mediumseagreen')
axes[1].set_title('Top 10 Titles in Final Top 100')
plt.tight_layout(); plt.show()
print('\nDone! Submit submission.csv')
