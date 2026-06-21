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
# # Notebook 1 — PRECOMPUTE
# **Run this once offline — no time limit.**
#
# What it does:
# - Loads all 100K candidates
# - Extracts structured features
# - Generates embeddings for all candidates + JD
# - Saves everything to disk as artifacts
#
# Output files saved to disk:
# - `features.parquet` — structured scores for all 100K candidates
# - `embeddings.npy` — embedding matrix (100K × 384)
# - `candidate_ids.npy` — aligned candidate IDs
# - `jd_embedding.npy` — JD embedding vector
#
# These files are then loaded by `RANK_notebook.ipynb` which runs in under 5 minutes.

# %% [markdown]
# ## Cell 1 — Install dependencies

# %%
# !pip install sentence-transformers pandas numpy pyarrow tqdm matplotlib -q

# %% [markdown]
# ## Cell 2 — Imports

# %%
import json, re, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, datetime
from tqdm.auto import tqdm
print('Imports OK')

# %% [markdown]
# ## Cell 3 — Config (change paths here)

# %%
# ── CHANGE THESE ──────────────────────────────────────────────────────────────
CANDIDATES_FILE = '/home/richa-kumari/Downloads/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl'   # path to your candidates file
ARTIFACTS_DIR   = '/home/richa-kumari/Documents/redrob/artifacts'          # folder where outputs will be saved

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # fast CPU model, 384-dim
EMBED_BATCH     = 64                   # reduce to 32 if memory errors
MAX_DOC_CHARS   = 1024                 # truncate candidate text

Path(ARTIFACTS_DIR).mkdir(exist_ok=True)
print(f'Candidates file exists: {Path(CANDIDATES_FILE).exists()}')
print(f'Artifacts will be saved to: {ARTIFACTS_DIR}/')

# %% [markdown]
# ## Cell 4 — All scoring logic

# %%
# ══════════════════════════════════════════════════════════════════════════════
# SCORING LOGIC (copy of features.py — post EDA v2)
# ══════════════════════════════════════════════════════════════════════════════

GOOD_TITLES = re.compile(
    r'\b(machine.?learning|ml.?engineer|ai.?engineer|applied.?scien|'
    r'nlp|search.?engineer|retrieval|ranking|recommendation|embedding|'
    r'llm|generative.?ai|inference|research.?engineer|data.?scien|'
    r'backend.?engineer|data.?engineer|software.?engineer|'
    r'senior.?engineer|lead.?engineer|principal.?engineer|staff.?engineer|'
    r'full.?stack|platform.?engineer|algorithm|ai.?specialist)\b',
    re.IGNORECASE)

BAD_TITLES = re.compile(
    r'\b(marketing|accountant|civil.?engineer|mechanical.?engineer|'
    r'graphic.?design|content.?writer|customer.?support|sales.?executive|'
    r'hr.?manager|human.?resource|recruiter|finance|legal|supply.?chain|'
    r'operations.?manager|project.?manager|qa.?engineer)\b',
    re.IGNORECASE)

SERVICES_COMPANIES = re.compile(
    r'\b(tcs|tata.?consultancy|infosys|wipro|accenture|cognizant|capgemini|'
    r'hcl|tech.?mahindra|mphasis|mindtree|hexaware|ltimindtree|'
    r'l&t.?infotech|niit|mastech|patni)\b',
    re.IGNORECASE)

PREFERRED_LOCATIONS = re.compile(
    r'\b(pune|noida|delhi|gurugram|gurgaon|ncr|hyderabad|mumbai|'
    r'bangalore|bengaluru|chennai|india)\b',
    re.IGNORECASE)

TECH_INDUSTRY = re.compile(
    r'\b(technology|software|saas|fintech|edtech|healthtech|ai|ml|'
    r'data|cloud|internet|e.?commerce|startup|product|food.?delivery|'
    r'conversational.?ai|adtech)\b',
    re.IGNORECASE)

CS_FIELDS = re.compile(
    r'\b(computer|software|information|data|electrical|electronics|'
    r'mathematics|statistics|machine.?learning|ai|engineering)\b',
    re.IGNORECASE)

CORE_AI_SKILLS = {
    'sentence-transformers','sentence_transformers','embeddings','embedding',
    'vector database','vector search','semantic search','hybrid search',
    'faiss','pinecone','weaviate','qdrant','milvus','opensearch',
    'elasticsearch','retrieval','ranking','reranking','re-ranking',
    'lora','qlora','peft','fine-tuning','fine tuning','fine-tuning llms',
    'rag','retrieval augmented generation',
    'bert','bge','e5','openai embeddings',
    'nlp','information retrieval','recommendation system',
    'recommendation engine','llm','large language model',
    'transformers','hugging face','hugging face transformers',
    'pytorch','tensorflow','xgboost','neural ranking',
}

KEYWORD_FLUFF = {
    'langchain','chatgpt','openai api','gpt-4','gpt4','chatbot',
    'prompt engineering','no-code','zapier','content writing',
    'agile','six sigma','sap','salesforce crm','illustrator',
}

PROFICIENCY_WEIGHT = {'beginner':0.10,'intermediate':0.45,'advanced':1.00,'expert':1.00}
TIER_SCORE = {'tier_1':1.0,'tier_2':0.75,'tier_3':0.50,'tier_4':0.25,'unknown':0.35}
WEIGHTS = {'title':0.25,'career':0.30,'skill':0.25,'experience':0.10,'location':0.05,'education':0.05}

def _days_since(date_str, ref=None):
    if not date_str: return 9999
    ref = ref or date.today()
    try: return (ref - datetime.strptime(date_str[:10],'%Y-%m-%d').date()).days
    except: return 9999

def _clamp(v, lo=0.0, hi=1.0): return max(lo, min(hi, v))
def _safe(v, default=0.0): return default if (v is None or v < 0) else float(v)

def honeypot_penalty(c):
    profile=c.get('profile',{}); career=c.get('career_history',[])
    skills=c.get('skills',[]); signals=c.get('redrob_signals',{}); flags=0
    stated=profile.get('years_of_experience',0) or 0
    summed=sum(j.get('duration_months',0) or 0 for j in career)/12
    if stated>0 and summed>0 and stated/max(summed,0.1)>3.0: flags+=2
    ez=[s for s in skills if s.get('proficiency')=='expert' and (s.get('duration_months') or 0)==0]
    flags+=2 if len(ez)>=2 else (1 if ez else 0)
    ec=sum(1 for s in skills if s.get('proficiency')=='expert')
    flags+=2 if ec>=6 else (1 if ec>=4 else 0)
    if signals.get('open_to_work_flag') and _days_since(signals.get('last_active_date'))>365: flags+=1
    if signals.get('recruiter_response_rate')==0 and signals.get('offer_acceptance_rate')==1.0: flags+=2
    if flags>=4: return 0.0
    if flags>=3: return 0.15
    if flags>=2: return 0.40
    if flags>=1: return 0.75
    return 1.0

def title_score(c):
    profile=c.get('profile',{}); career=c.get('career_history',[])
    titles=[profile.get('current_title','')] + [j.get('title','') for j in career]
    combined=' '.join(titles)
    good=len(GOOD_TITLES.findall(combined)); bad=len(BAD_TITLES.findall(combined))
    cur=profile.get('current_title','')
    if GOOD_TITLES.search(cur): good+=2
    if BAD_TITLES.search(cur): bad+=2
    return _clamp(good*0.25 - bad*0.30)

def career_score(c):
    career=c.get('career_history',[])
    if not career: return 0.0
    total=ai=services=0; tenures=[]
    for job in career:
        dur=job.get('duration_months') or 0; total+=dur; tenures.append(dur)
        company=job.get('company',''); industry=job.get('industry','')
        title=job.get('title',''); desc=(job.get('description') or '').lower()
        is_svc=bool(SERVICES_COMPANIES.search(company))
        is_ai=bool(GOOD_TITLES.search(title)) or bool(TECH_INDUSTRY.search(industry))
        has_ai=any(kw in desc for kw in ['embedding','retrieval','ranking','nlp',
            'machine learning','vector','search','recommendation','llm','fine-tun','transformer'])
        if is_svc: services+=dur
        elif is_ai or has_ai: ai+=dur
    if total==0: return 0.0
    stability=_clamp((sum(tenures)/len(tenures)-12)/24)
    return _clamp(ai/total*0.55 + (1-services/total)*0.25 + stability*0.20)

def skill_score(c):
    skills=c.get('skills',[])
    assessments=(c.get('redrob_signals') or {}).get('skill_assessment_scores') or {}
    ai_score=0.0; fluff=0
    for s in skills:
        name=(s.get('name') or '').lower()
        prof=PROFICIENCY_WEIGHT.get(s.get('proficiency','beginner'),0.10)
        dur=_clamp(0.3+(s.get('duration_months') or 0)/28)
        end=1.0+min(s.get('endorsements') or 0,50)/125
        if any(core in name for core in CORE_AI_SKILLS):
            bonus=next((min(_safe(av,0)/333,0.30) for ak,av in assessments.items()
                       if ak.lower() in name or name in ak.lower()),0.0)
            ai_score+=prof*dur*end+bonus
        if any(f in name for f in KEYWORD_FLUFF): fluff+=1
    return _clamp(ai_score/10.0)*_clamp(1.0-fluff*0.08)

def experience_score(c):
    yoe=c.get('profile',{}).get('years_of_experience') or 0
    if yoe<2: return 0.10
    if yoe<4: return 0.35+(yoe-2)*0.10
    if yoe<=9: return 0.60+(yoe-4)*0.06
    if yoe<=12: return 0.88-(yoe-9)*0.04
    return 0.70

def location_score(c):
    profile=c.get('profile',{}); signals=c.get('redrob_signals',{})
    loc=(profile.get('location','') + ' ' + profile.get('country','')).strip()
    if PREFERRED_LOCATIONS.search(loc): return 1.0
    if signals.get('willing_to_relocate'): return 0.60
    if 'india' not in loc.lower(): return 0.25
    return 0.50

def education_score(c):
    edu=c.get('education',[])
    if not edu: return 0.30
    best=0.0
    for e in edu:
        t=TIER_SCORE.get(e.get('tier','unknown'),0.35)
        f=0.15 if CS_FIELDS.search(e.get('field_of_study') or '') else 0.0
        best=max(best,t+f)
    return _clamp(best)

def availability_multiplier(c):
    signals=c.get('redrob_signals',{})
    recency=_clamp(1.0-_days_since(signals.get('last_active_date'))/200)
    otw=1.0 if signals.get('open_to_work_flag') else 0.45
    rrr=_safe(signals.get('recruiter_response_rate'),0.3)
    icr=_safe(signals.get('interview_completion_rate'),0.5)
    np_d=signals.get('notice_period_days') or 90
    notice=1.0 if np_d<=30 else(0.85 if np_d<=60 else(0.65 if np_d<=90 else(0.40 if np_d<=120 else 0.20)))
    gh_raw=signals.get('github_activity_score')
    github=0.50 if(gh_raw is None or gh_raw<0) else _clamp(gh_raw/100)
    mult=recency*0.28+otw*0.25+rrr*0.20+icr*0.15+notice*0.07+github*0.05
    return _clamp(mult,0.20,1.0)

def structured_score(c):
    t,cr,s,e,l,ed=(title_score(c),career_score(c),skill_score(c),
                   experience_score(c),location_score(c),education_score(c))
    composite=(t*WEIGHTS['title']+cr*WEIGHTS['career']+s*WEIGHTS['skill']+
               e*WEIGHTS['experience']+l*WEIGHTS['location']+ed*WEIGHTS['education'])
    return {'title_score':round(t,4),'career_score':round(cr,4),'skill_score':round(s,4),
            'experience_score':round(e,4),'location_score':round(l,4),
            'education_score':round(ed,4),'structured':round(composite,4)}

def build_candidate_doc(c):
    parts=[]; p=c.get('profile',{})
    parts.append(p.get('headline') or '')
    parts.append(p.get('summary') or '')
    parts.append(f"Current: {p.get('current_title','')} at {p.get('current_company','')} ({p.get('current_industry','')})")
    for job in (c.get('career_history') or [])[:5]:
        parts.append(f"{job.get('title','')} at {job.get('company','')} ({job.get('industry','')}): {job.get('description','')}")
    top_skills=[s.get('name','') for s in (c.get('skills') or []) if s.get('proficiency') in ('advanced','expert')]
    if top_skills: parts.append('Advanced skills: '+', '.join(top_skills))
    return ' | '.join(pt for pt in parts if pt.strip())[:MAX_DOC_CHARS]

print('All scoring functions defined OK')

# %% [markdown]
# ## Cell 5 — Load all 100K candidates

# %%
t0=time.time(); candidates=[]
with open(CANDIDATES_FILE,'r',encoding='utf-8') as f:
    for line in tqdm(f,desc='Loading',total=100_000):
        line=line.strip()
        if line: candidates.append(json.loads(line))
print(f'Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s')


# %% [markdown]
# ## Cell 6 — Extract structured features → save features.parquet

# %%
# We also store every field needed for reasoning strings here,
# so RANK_notebook.ipynb NEVER has to scan candidates.jsonl at rank-time.
# This is what keeps the ranking step's wall-clock time low and robust.

def _top_ai_skills(c, n=3):
    skills = c.get('skills', [])
    hits = [s.get('name','') for s in skills
            if any(core in (s.get('name') or '').lower() for core in CORE_AI_SKILLS)
            and s.get('proficiency') in ('advanced','expert')]
    return ', '.join(hits[:n]) if hits else 'no core AI skills'

t0 = time.time()
rows = []
for c in tqdm(candidates, desc='Extracting features'):
    comps   = structured_score(c)
    hp      = honeypot_penalty(c)
    av      = availability_multiplier(c)
    p       = c['profile']
    signals = c.get('redrob_signals', {})
    rows.append({
        'candidate_id':      c['candidate_id'],
        **comps,
        'honeypot_mult':     round(hp, 4),
        'availability_mult': round(av, 4),
        'base_score':        round(comps['structured'] * hp * av, 4),
        'yoe':               p.get('years_of_experience', 0),
        'current_title':     p.get('current_title', ''),
        'current_company':   p.get('current_company', ''),
        'location':          p.get('location', '') or p.get('country', '') or 'Unknown',
        'notice_days':       signals.get('notice_period_days', 90),
        'response_rate':     signals.get('recruiter_response_rate', -1),
        'open_to_work':      signals.get('open_to_work_flag', False),
        'github_score':      signals.get('github_activity_score', -1),
        'top_ai_skills':     _top_ai_skills(c),
    })

features_df = pd.DataFrame(rows)
features_df.to_parquet(f'{ARTIFACTS_DIR}/features.parquet', index=False)
print(f'Features extracted in {time.time()-t0:.1f}s -- saved features.parquet')
print(f'Shape: {features_df.shape}')
print('Reasoning fields included -- RANK notebook will not need candidates.jsonl')

# %% [markdown]
# ## Cell 7 — Build text documents for embedding

# %%
t0=time.time()
docs=[build_candidate_doc(c) for c in tqdm(candidates,desc='Building docs')]
print(f'Built {len(docs):,} docs in {time.time()-t0:.1f}s')

# %% [markdown]
# ## Cell 8 — Load model and embed the JD → save jd_embedding.npy

# %%
from sentence_transformers import SentenceTransformer
print(f'Loading {EMBEDDING_MODEL} ...')
model=SentenceTransformer(EMBEDDING_MODEL)

JD_TEXT="""
Senior AI Engineer founding team Redrob AI talent intelligence platform Pune Noida India hybrid.
5 to 9 years experience applied machine learning production systems.
Production embeddings retrieval sentence transformers BGE E5 OpenAI embeddings deployed real users.
Vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS.
Ranking retrieval evaluation NDCG MRR MAP A/B testing. LLM fine-tuning LoRA QLoRA PEFT.
Learning to rank XGBoost neural ranking. Semantic search NLP information retrieval recommendation systems.
Product company Swiggy Zomato BYJU's Dream11 Meesho Razorpay Yellow.ai not consulting TCS Infosys Wipro.
Strong Python. Ship working systems not demos. Applied ML engineer.
"""

jd_emb=model.encode([JD_TEXT.strip()],normalize_embeddings=True)[0]
np.save(f'{ARTIFACTS_DIR}/jd_embedding.npy',jd_emb)
print(f'JD embedded and saved — shape: {jd_emb.shape}')

# %% [markdown]
# ## Cell 9 — Embed all 100K candidates → save embeddings.npy + candidate_ids.npy
# **This is the slow cell — ~45-90 min on CPU. Leave it running.**

# %%
n=len(docs); dim=jd_emb.shape[0]
all_embs=np.zeros((n,dim),dtype=np.float32)
n_batches=math.ceil(n/EMBED_BATCH)
print(f'Embedding {n:,} candidates in {n_batches} batches of {EMBED_BATCH} ...')
print('Leave this running — go grab a coffee ☕')
t0=time.time()
for i in tqdm(range(n_batches),desc='Embedding'):
    start=i*EMBED_BATCH; end=min(start+EMBED_BATCH,n)
    all_embs[start:end]=model.encode(docs[start:end],normalize_embeddings=True,show_progress_bar=False)
print(f'Done in {(time.time()-t0)/60:.1f} min')

# Save embeddings and aligned IDs
candidate_ids=np.array([c['candidate_id'] for c in candidates])
np.save(f'{ARTIFACTS_DIR}/embeddings.npy',all_embs)
np.save(f'{ARTIFACTS_DIR}/candidate_ids.npy',candidate_ids)
print(f'Saved embeddings.npy {all_embs.shape} and candidate_ids.npy')

# %% [markdown]
# ## Cell 10 — Verify all artifacts saved correctly

# %%
import os
print('Artifacts saved:')
for fname in ['features.parquet','embeddings.npy','candidate_ids.npy','jd_embedding.npy']:
    fpath=f'{ARTIFACTS_DIR}/{fname}'
    size=os.path.getsize(fpath)/1024/1024
    print(f'  ✓ {fname:30s} {size:.1f} MB')
print()
print('PRECOMPUTE COMPLETE.')
print('Now run RANK_notebook.ipynb to produce submission.csv in under 5 minutes.')
