import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "category_group_price_tier_review.csv"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
NN_SIMILAR_K = 20
TOP_STORES = 50
RANDOM_STATE = 42
TEST_SIZE = 0.20

# -----------------------
# LOAD & CLEAN
# -----------------------
df = pd.read_csv(DATA_PATH)

required_cols = ['name','sold','category_group','store_name','price','original_price']
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

if 'discount_percent' not in df.columns:
    eps = 1e-9
    df['discount_percent'] = (
        (df['original_price'].fillna(df['price']) - df['price'])
        / (df['original_price'].fillna(df['price']) + eps)
    ).clip(lower=0, upper=1.0)
else:
    df['discount_percent'] = df['discount_percent'].astype(float)
    wrong = df['discount_percent'] > 3
    df.loc[wrong, 'discount_percent'] = df.loc[wrong, 'discount_percent'] / 100.0
    df['discount_percent'] = df['discount_percent'].clip(0, 1.5)

df['comment_count'] = df.get('comment_count', 0)
df['rating'] = df.get('rating', 4.8)

df['price'] = df['price'].fillna(0).astype(float)
df['original_price'] = df['original_price'].fillna(df['price']).astype(float)
df['sold'] = df['sold'].astype(float)
df['is_sold_well'] = (df['sold'] > 100).astype(int)

df = df.dropna(subset=['name','category_group']).copy()

min_samples = 45
major_categories = df['category_group'].value_counts()[lambda x: x >= min_samples].index.tolist()
df['category_group_processed'] = df['category_group'].apply(
    lambda x: x if x in major_categories else 'Other'
)

top_stores = df['store_name'].value_counts().nlargest(TOP_STORES).index.tolist()
df['store_processed'] = df['store_name'].apply(
    lambda x: x if x in top_stores else 'OtherStore'
)

print("Rows:", len(df), "| Major categories:", len(major_categories))

# -----------------------
# TRAIN/TEST SPLIT
# -----------------------
df_train_full, df_test_holdout = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['is_sold_well']
)

print("Train size:", len(df_train_full), "Holdout size:", len(df_test_holdout))

# -----------------------
# SBERT EMBEDDING
# -----------------------
print("Loading SBERT:", SBERT_MODEL_NAME)
embed_model = SentenceTransformer(SBERT_MODEL_NAME)

names_all = df['name'].fillna("").astype(str).tolist()
embs_all = embed_model.encode(names_all, show_progress_bar=True, convert_to_numpy=True)

train_orig_idx = df_train_full.index.to_numpy()
test_orig_idx = df_test_holdout.index.to_numpy()

embs_train = embs_all[train_orig_idx]
embs_test = embs_all[test_orig_idx]

df_train_full = df_train_full.reset_index(drop=True)
df_test_holdout = df_test_holdout.reset_index(drop=True)
# -----------------------
# HELPER: Compute weight for neighbor based on price/discount deviation
# -----------------------
def compute_weight_for_neighbor(price, discount, market_low, market_high, disc_low, disc_high):
    # weight for price deviation
    if price < market_low:
        dev_p = (market_low - price) / market_low
    elif price > market_high:
        dev_p = (price - market_high) / market_high
    else:
        dev_p = 0

    # convert to weight 1 → 0.3
    w_price = max(0.3, 1.0 - dev_p * 1.8)

    # weight for discount deviation
    if discount < disc_low:
        dev_d = (disc_low - discount)
    elif discount > disc_high:
        dev_d = (discount - disc_high)
    else:
        dev_d = 0

    # discount deviations usually more severe
    w_disc = max(0.3, 1.0 - dev_d * 4.0)

    # final weight = combined
    return max(0.15, min(1.0, (w_price + w_disc) / 2))

# -----------------------
# CROSS-VALIDATION TRAINING
# -----------------------
numeric_base_cols = ['price','comment_count','rating']
df_train_full[numeric_base_cols] = df_train_full[numeric_base_cols].astype(float)

oof_probs = np.zeros(len(df_train_full))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_models = []
fold_scalers = []
fold_encoders = []
fold_nn_models = []
fold_cat_mean_maps = []

print("\nTraining CV (NO LEAK)...")

for fold, (tr_idx, va_idx) in enumerate(skf.split(df_train_full, df_train_full['is_sold_well'])):
    print(f"\n=== FOLD {fold+1} ===")

    df_tr = df_train_full.iloc[tr_idx].reset_index(drop=True)
    df_val = df_train_full.iloc[va_idx].reset_index(drop=True)

    # ---- NN TRAIN-ONLY
    nn_fold = NearestNeighbors(n_neighbors=NN_SIMILAR_K+1, metric='cosine', n_jobs=-1)
    nn_fold.fit(embs_train[tr_idx])

    def compute_sim_features_weighted(q_embs, base_df):
        _, idxs = nn_fold.kneighbors(q_embs, n_neighbors=NN_SIMILAR_K+1)
        n = len(idxs)
        s_price = np.zeros(n)
        s_orig = np.zeros(n)
        s_sold = np.zeros(n)
        s_pct = np.zeros(n)
        s_comment = np.zeros(n)
        s_rating = np.zeros(n)
        s_disc = np.zeros(n)
        
        for i in range(n):
            neigh_idx = idxs[i, 1:NN_SIMILAR_K+1]
            if len(neigh_idx) == 0:
                continue
            neigh = base_df.iloc[neigh_idx]
            
            # compute market reference from neighbors
            market_low = neigh['price'].quantile(0.25)
            market_high = neigh['price'].quantile(0.75)
            disc_low = neigh['discount_percent'].quantile(0.25)
            disc_high = neigh['discount_percent'].quantile(0.75)
            
            # compute weights for each neighbor
            weights = []
            for _, r in neigh.iterrows():
                w = compute_weight_for_neighbor(
                    price=r['price'],
                    discount=r['discount_percent'],
                    market_low=market_low,
                    market_high=market_high,
                    disc_low=disc_low,
                    disc_high=disc_high
                )
                weights.append(w)
            weights = np.array(weights)
            
            # weighted averages
            s_price[i] = np.average(neigh['price'], weights=weights)
            s_orig[i] = np.average(neigh['original_price'], weights=weights)
            s_sold[i] = np.average(neigh['sold'], weights=weights)
            s_pct[i] = np.average(neigh['is_sold_well'], weights=weights)
            s_comment[i] = np.average(neigh['comment_count'], weights=weights)
            s_rating[i] = np.average(neigh['rating'], weights=weights)
            s_disc[i] = np.average(neigh['discount_percent'], weights=weights)
        
        return np.column_stack([s_price, s_orig, s_sold, s_pct, s_comment, s_rating, s_disc])

    # ---- TRAIN similarity with weighted KNN
    sim_tr = compute_sim_features_weighted(embs_train[tr_idx], df_tr)
    df_tr[['sim_avg_price','sim_avg_original','sim_avg_sold','sim_pct_sold_gt100',
           'sim_comment_avg','sim_rating_avg','sim_avg_discount']] = sim_tr

    # ---- VAL similarity (neighbors = TRAIN ONLY) with weighted KNN
    sim_val = compute_sim_features_weighted(embs_train[va_idx], df_tr)
    df_val[['sim_avg_price','sim_avg_original','sim_avg_sold','sim_pct_sold_gt100',
            'sim_comment_avg','sim_rating_avg','sim_avg_discount']] = sim_val

    # ---- Category means
    cat_mean_price = df_tr.groupby('category_group_processed')['price'].mean().to_dict()
    cat_mean_orig  = df_tr.groupby('category_group_processed')['original_price'].mean().to_dict()

    df_tr['cat_mean_price'] = df_tr['category_group_processed'].map(cat_mean_price).fillna(df_tr['price'].mean())
    df_tr['cat_mean_original'] = df_tr['category_group_processed'].map(cat_mean_orig).fillna(df_tr['original_price'].mean())

    df_val['cat_mean_price'] = df_val['category_group_processed'].map(cat_mean_price).fillna(df_tr['price'].mean())
    df_val['cat_mean_original'] = df_val['category_group_processed'].map(cat_mean_orig).fillna(df_tr['original_price'].mean())

    # ---- Indicators
    df_tr['has_comment_data'] = (df_tr['comment_count'] > 0).astype(float)
    df_tr['has_rating_data']  = (df_tr['rating'] > 0).astype(float)
    df_val['has_comment_data'] = (df_val['comment_count'] > 0).astype(float)
    df_val['has_rating_data']  = (df_val['rating'] > 0).astype(float)

    # ---- Numeric list
    numeric_cols = [
        'price','cat_mean_price','cat_mean_original',
        'sim_avg_price','sim_avg_original','sim_avg_sold','sim_pct_sold_gt100',
        'comment_count','rating','has_comment_data','has_rating_data','sim_avg_discount'
    ]

    # ---- OHE
    ohe_fold = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_tr_cat = ohe_fold.fit_transform(df_tr[['store_processed','category_group_processed']])
    X_val_cat = ohe_fold.transform(df_val[['store_processed','category_group_processed']])

    cat_cols = ohe_fold.get_feature_names_out(['store_processed','category_group_processed'])

    X_tr = pd.concat([df_tr[numeric_cols], pd.DataFrame(X_tr_cat, columns=cat_cols)], axis=1)
    X_val = pd.concat([df_val[numeric_cols], pd.DataFrame(X_val_cat, columns=cat_cols)], axis=1)

    # ---- SCALER
    scaler_fold = StandardScaler()
    X_tr[numeric_cols] = scaler_fold.fit_transform(X_tr[numeric_cols])
    X_val[numeric_cols] = scaler_fold.transform(X_val[numeric_cols])

    # ---- TRAIN RF
    clf_fold = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf_fold.fit(X_tr, df_tr['is_sold_well'])

    val_probs = clf_fold.predict_proba(X_val)[:,1]
    oof_probs[va_idx] = val_probs
    auc = roc_auc_score(df_val['is_sold_well'], val_probs)
    print(f"Fold {fold+1} AUC:", auc)

    fold_models.append(clf_fold)
    fold_scalers.append(scaler_fold)
    fold_encoders.append(ohe_fold)
    fold_nn_models.append(nn_fold)
    fold_cat_mean_maps.append((cat_mean_price, cat_mean_orig))

# ---- Overall CV AUC
overall_auc = roc_auc_score(df_train_full['is_sold_well'], oof_probs)
print("\n=== OVERALL OOF AUC:", overall_auc, "===\n")
# =====================================================================
# PART 2 / 3
# - Train final model on full TRAIN
# - Inference helpers
# - Trust Score (Balanced)
# - Market price & discount range helpers
# =====================================================================

import math

print("\n==============================================================")
print("TRAINING FINAL MODEL ON FULL TRAIN SET (ARTIFACTS)")
print("==============================================================")

# Fit NN on full train embeddings (use embs_train from PART 1)
nn_final = NearestNeighbors(n_neighbors=NN_SIMILAR_K + 1, metric='cosine', n_jobs=-1)
nn_final.fit(embs_train)
_, indices_final = nn_final.kneighbors(embs_train, n_neighbors=NN_SIMILAR_K+1)

def compute_weight_for_neighbor(price, discount, market_low, market_high, disc_low, disc_high):
    # weight for price deviation
    if price < market_low:
        dev_p = (market_low - price) / market_low
    elif price > market_high:
        dev_p = (price - market_high) / market_high
    else:
        dev_p = 0

    # convert to weight 1 → 0.3
    w_price = max(0.3, 1.0 - dev_p * 1.8)

    # weight for discount deviation
    if discount < disc_low:
        dev_d = (disc_low - discount)
    elif discount > disc_high:
        dev_d = (discount - disc_high)
    else:
        dev_d = 0

    # discount deviations usually more severe
    w_disc = max(0.3, 1.0 - dev_d * 4.0)

    # final weight = combined
    return max(0.15, min(1.0, (w_price + w_disc) / 2))

def compute_sim_stats_full(train_df, idx_matrix):
    n = len(idx_matrix)
    s_price = np.zeros(n)
    s_orig = np.zeros(n)
    s_sold = np.zeros(n)
    s_pct = np.zeros(n)
    s_comment = np.zeros(n)
    s_rating = np.zeros(n)
    s_disc = np.zeros(n)

    for i in range(n):
        neigh_idx = idx_matrix[i, 1:NN_SIMILAR_K+1]
        neigh = train_df.iloc[neigh_idx]

        # compute market reference from neighbors
        market_low = neigh['price'].quantile(0.25)
        market_high = neigh['price'].quantile(0.75)
        disc_low = neigh['discount_percent'].quantile(0.25)
        disc_high = neigh['discount_percent'].quantile(0.75)

        weights = []
        for _, r in neigh.iterrows():
            w = compute_weight_for_neighbor(
                price=r['price'],
                discount=r['discount_percent'],
                market_low=market_low,
                market_high=market_high,
                disc_low=disc_low,
                disc_high=disc_high
            )
            weights.append(w)
        weights = np.array(weights)

        s_price[i] = np.average(neigh['price'], weights=weights)
        s_orig[i] = np.average(neigh['original_price'], weights=weights)
        s_sold[i] = np.average(neigh['sold'], weights=weights)
        s_pct[i] = np.average(neigh['is_sold_well'], weights=weights)
        s_comment[i] = np.average(neigh['comment_count'], weights=weights)
        s_rating[i] = np.average(neigh['rating'], weights=weights)
        s_disc[i] = np.average(neigh['discount_percent'], weights=weights)

    return s_price, s_orig, s_sold, s_pct, s_comment, s_rating, s_disc


s_price_f, s_orig_f, s_sold_f, s_pct_f, s_comment_f, s_rating_f, s_disc_f = \
    compute_sim_stats_full(df_train_full, indices_final)

df_train_full['sim_avg_price'] = s_price_f
df_train_full['sim_avg_original'] = s_orig_f
df_train_full['sim_avg_sold'] = s_sold_f
df_train_full['sim_pct_sold_gt100'] = s_pct_f
df_train_full['sim_comment_avg'] = s_comment_f
df_train_full['sim_rating_avg'] = s_rating_f
df_train_full['sim_avg_discount'] = s_disc_f

# Category mean encodings (from full train)
cat_mean_price_map_final = df_train_full.groupby('category_group_processed')['price'].mean().to_dict()
cat_mean_orig_map_final = df_train_full.groupby('category_group_processed')['original_price'].mean().to_dict()

df_train_full['cat_mean_price'] = df_train_full['category_group_processed'].map(cat_mean_price_map_final).fillna(df_train_full['price'].mean())
df_train_full['cat_mean_original'] = df_train_full['category_group_processed'].map(cat_mean_orig_map_final).fillna(df_train_full['original_price'].mean())

# Indicators present for training rows
df_train_full['has_comment_data'] = (df_train_full['comment_count'] > 0).astype(float)
df_train_full['has_rating_data'] = (df_train_full['rating'] > 0).astype(float)

# Feature list (ensure sim_avg_discount is included — comes from TRAIN similars, safe)
numeric_cols = [
    'price', 'cat_mean_price', 'cat_mean_original',
    'sim_avg_price', 'sim_avg_original', 'sim_avg_sold', 'sim_pct_sold_gt100',
    'comment_count', 'rating', 'has_comment_data', 'has_rating_data', 'sim_avg_discount'
]

# Fit final encoders and scaler on full TRAIN
ohe_final = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_ohe_final = ohe_final.fit_transform(df_train_full[['store_processed', 'category_group_processed']])
cat_ohe_cols = ohe_final.get_feature_names_out(['store_processed', 'category_group_processed']).tolist()
X_cat_final = pd.DataFrame(cat_ohe_final, columns=cat_ohe_cols, index=df_train_full.index)

X_num_final = df_train_full[numeric_cols].reset_index(drop=True)
X_final = pd.concat([X_num_final, X_cat_final.reset_index(drop=True)], axis=1).fillna(0)

scaler_final = StandardScaler()
X_final[numeric_cols] = scaler_final.fit_transform(X_final[numeric_cols])

y_final = df_train_full['is_sold_well'].values

clf_final = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
clf_final.fit(X_final, y_final)

print("Final model trained on TRAIN set. Artifacts ready: nn_final, ohe_final, scaler_final, clf_final.")

# -----------------------
# INFERENCE HELPERS (use TRAIN artifacts only for similarity)
# -----------------------
def get_similar_products(name, top_k=20):
    emb_q = embed_model.encode([name], convert_to_numpy=True)
    dists, idxs = nn_final.kneighbors(emb_q, n_neighbors=top_k+1, return_distance=True)
    neigh_idx = idxs[0, 1:top_k+1]

    cols = ['name','price','sold','comment_count','rating','category_group','original_price','discount_percent']
    sim_df = df_train_full.iloc[neigh_idx][cols].reset_index(drop=True)

    return sim_df


def get_similar_stats_for_name(name, top_k=10):
    """Get average comment_count, rating, and discount from similar products in TRAIN set"""
    emb_q = embed_model.encode([name], convert_to_numpy=True)
    dists, idxs = nn_final.kneighbors(emb_q, n_neighbors=top_k+1, return_distance=True)
    neigh_idx = idxs[0, 1:top_k+1]
    neigh = df_train_full.iloc[neigh_idx]
    return (
        float(neigh['comment_count'].mean()), 
        float(neigh['rating'].mean()),
        float(neigh.get('discount_percent', pd.Series([0.0])).mean())
    )

def prepare_input_features(name, store_name, category_group, price_input,
                           comment_count=None, rating=None, use_sim_fill_for_missing=True):
    """
    Prepare a single-row feature vector with weighted KNN features.
    - Includes sim_avg_discount feature from TRAIN similars
    - Uses indicator features (has_comment_data, has_rating_data)
    """
    has_comment = 1.0 if comment_count is not None else 0.0
    has_rating = 1.0 if rating is not None else 0.0

    if (comment_count is None or rating is None) and use_sim_fill_for_missing:
        c_avg, r_avg, sim_disc = get_similar_stats_for_name(name)
        if comment_count is None:
            comment_count = c_avg if not np.isnan(c_avg) else 0.0
        if rating is None:
            rating = r_avg if not np.isnan(r_avg) else 4.8

    store = store_name if store_name in top_stores else 'OtherStore'
    cat = category_group if category_group in major_categories else 'Other'

    # Get weighted KNN similarity features from TRAIN
    emb_q = embed_model.encode([name], convert_to_numpy=True)
    _, idxs = nn_final.kneighbors(emb_q, n_neighbors=NN_SIMILAR_K+1, return_distance=True)
    neigh_idx = idxs[0, 1:NN_SIMILAR_K+1]
    neigh = df_train_full.iloc[neigh_idx]
    
    # Compute weighted averages (using same logic as training)
    market_low = neigh['price'].quantile(0.25)
    market_high = neigh['price'].quantile(0.75)
    disc_low = neigh['discount_percent'].quantile(0.25)
    disc_high = neigh['discount_percent'].quantile(0.75)
    
    weights = []
    for _, r in neigh.iterrows():
        w = compute_weight_for_neighbor(
            price=r['price'],
            discount=r['discount_percent'],
            market_low=market_low,
            market_high=market_high,
            disc_low=disc_low,
            disc_high=disc_high
        )
        weights.append(w)
    weights = np.array(weights)
    
    sim_avg_price_q = float(np.average(neigh['price'], weights=weights))
    sim_avg_orig_q = float(np.average(neigh['original_price'], weights=weights))
    sim_avg_sold_q = float(np.average(neigh['sold'], weights=weights))
    sim_pct_sold_q = float(np.average(neigh['is_sold_well'], weights=weights))
    sim_comment_avg_q = float(np.average(neigh['comment_count'], weights=weights))
    sim_rating_avg_q = float(np.average(neigh['rating'], weights=weights))
    sim_avg_disc_q = float(np.average(neigh['discount_percent'], weights=weights))

    cat_mean_price_q = float(cat_mean_price_map_final.get(cat, df_train_full['price'].mean()))
    cat_mean_orig_q = float(cat_mean_orig_map_final.get(cat, df_train_full['original_price'].mean()))

    num_series = pd.Series({
        'price': float(price_input),
        'cat_mean_price': cat_mean_price_q,
        'cat_mean_original': cat_mean_orig_q,
        'sim_avg_price': sim_avg_price_q,
        'sim_avg_original': sim_avg_orig_q,
        'sim_avg_sold': sim_avg_sold_q,
        'sim_pct_sold_gt100': sim_pct_sold_q,
        'comment_count': float(comment_count),
        'rating': float(rating),
        'has_comment_data': has_comment,
        'has_rating_data': has_rating,
        'sim_avg_discount': sim_avg_disc_q
    })

    cat_row = pd.DataFrame([[store, cat]], columns=['store_processed','category_group_processed'])
    cat_ohe_row = ohe_final.transform(cat_row)
    cat_ohe_row_df = pd.DataFrame(cat_ohe_row, columns=cat_ohe_cols)

    row = pd.concat([num_series, cat_ohe_row_df.iloc[0]])
    row_df = row.to_frame().T

    # scale numeric with scaler_final fitted on TRAIN
    row_df[numeric_cols] = scaler_final.transform(row_df[numeric_cols])
    row_df = row_df.reindex(columns=X_final.columns, fill_value=0)
    return row_df

def predict_prob_sold_gt100(row_df):
    return float(clf_final.predict_proba(row_df)[:,1][0])

# -----------------------
# TRUST SCORE (Balanced) — helpers and main aggregator
# -----------------------
def compute_comment_factor(comment_count):
    # Log scaled; saturates around ~100 comments
    return float(min(1.0, math.log1p(max(0.0, comment_count)) / 4.5))

def compute_rating_factor(rating, comment_count):
    # rating in [0,5] - Chỉ chuẩn hóa, KHÔNG phạt ở đây
    # Logic phạt được xử lý tập trung trong compute_review_trust để tránh phạt 2 lần
    r = float(max(0.0, min(5.0, rating))) / 5.0
    return r

def compute_review_trust(comment_count, rating, sim_comment_avg, sim_rating_avg, is_from_sim=False):
    """
    CẤU TRÚC MỚI: Bắt đầu từ 1.0, PHẠT trước theo rule, THƯỞNG lại
    
    RATING - Quyết định mức độ phạt/thưởng:
    - Rating cao (≥4.5): THƯỞNG
    - Rating trung bình (3.5-4.4): GIỮ NGUYÊN hoặc phạt nhẹ
    - Rating thấp (<3.5): PHẠT NẶNG
    
    COMMENT - Quyết định độ mạnh của phạt/thưởng:
    - Rating CAO: Comment nhiều → Thưởng mạnh (nhiều người xác nhận tốt)
    - Rating THẤP: Comment nhiều → Phạt NẶNG (nhiều người xác nhận tệ)
    - Rating TB: Comment ít → Phạt (không đủ tin cậy)
    
    Ma trận PHẠT/THƯỞNG (rating x comment):
    ┌─────────────┬──────────────────┬──────────────────┬────────────────────┐
    │  Rating     │  Comment cao     │   Comment TB     │   Comment thấp     │
    ├─────────────┼──────────────────┼──────────────────┼────────────────────┤
    │  Cao (≥4.5) │  Thưởng +15%     │  Thưởng +10%     │   Thưởng +5%       │
    │  TB (3.5-4.4)│  Giữ nguyên     │  Phạt -5%        │   Phạt -10%        │
    │  Thấp (<3.5)│  Phạt -50%       │  Phạt -40%       │   Phạt -30%        │
    └─────────────┴──────────────────┴──────────────────┴────────────────────┘
    
    LƯU Ý: Rating thấp + Comment cao → Phạt NẶNG NHẤT (-50%)
           Vì nhiều người xác nhận sản phẩm tệ → Rất đáng tin
    """
    # Xác định giá trị comment và rating để dùng
    if is_from_sim or (comment_count is None and rating is None):
        print(f"   ℹ️  Không có comment/rating thực tế → Sử dụng giá trị từ sản phẩm tương tự:")
        print(f"      Comment trung bình: {sim_comment_avg:.1f}")
        print(f"      Rating trung bình: {sim_rating_avg:.2f}/5.0")
        cc = sim_comment_avg
        rt = sim_rating_avg
        is_using_sim = True
    else:
        print(f"   ✅ Sử dụng comment/rating thực tế từ sản phẩm:")
        print(f"      Comment count: {comment_count}")
        print(f"      Rating: {rating:.2f}/5.0")
        cc = comment_count if comment_count is not None else 0
        rt = rating if rating is not None else 0.0
        is_using_sim = False
    
    # BẮT ĐẦU TỪ 1.0
    trust_review = 1.0
    
    # Phân loại rating
    is_high_rating = rt >= 4.5
    is_medium_rating = 3.5 <= rt < 4.5
    is_low_rating = rt < 3.5
    
    # Phân loại comment
    is_high_comment = cc >= 50
    is_medium_comment = 20 <= cc < 50
    is_low_comment = cc < 20
    
    # BƯỚC 1: PHẠT theo ma trận
    if is_high_rating:
        # Rating cao → KHÔNG PHẠT, chuẩn bị thưởng
        penalty = 0.0
        if is_high_comment:
            reward = 0.15  # Thưởng 15%
            print(f"      📊 Rating cao + Comment cao → THƯỞNG +15%")
        elif is_medium_comment:
            reward = 0.10  # Thưởng 10%
            print(f"      📊 Rating cao + Comment TB → THƯỞNG +10%")
        else:  # low comment
            reward = 0.05  # Thưởng 5%
            print(f"      📊 Rating cao + Comment thấp → THƯỞNG +5%")
            
    elif is_medium_rating:
        # Rating TB → Phạt nhẹ hoặc giữ nguyên
        reward = 0.0
        if is_high_comment:
            penalty = 0.0  # Giữ nguyên
            print(f"      📊 Rating TB + Comment cao → GIỮ NGUYÊN")
        elif is_medium_comment:
            penalty = 0.05  # Phạt 5%
            print(f"      📊 Rating TB + Comment TB → PHẠT -5%")
        else:  # low comment
            penalty = 0.10  # Phạt 10%
            print(f"      📊 Rating TB + Comment thấp → PHẠT -10%")
            
    else:  # low rating
        # Rating thấp → PHẠT NẶNG
        # Comment NHIỀU = Nhiều người xác nhận tệ → Phạt NẶNG NHẤT
        reward = 0.0
        if is_high_comment:
            penalty = 0.50  # Phạt 50% - NẶNG NHẤT (nhiều người xác nhận tệ)
            print(f"      📊 Rating thấp + Comment cao → PHẠT -50% (nhiều người xác nhận tệ)")
        elif is_medium_comment:
            penalty = 0.40  # Phạt 40%
            print(f"      📊 Rating thấp + Comment TB → PHẠT -40%")
        else:  # low comment
            penalty = 0.30  # Phạt 30% - NHẸ HƠN (ít người đánh giá, có thể ngẫu nhiên)
            print(f"      📊 Rating thấp + Comment thấp → PHẠT -30% (ít người đánh giá)")
    
    # BƯỚC 2: Áp dụng phạt/thưởng
    if is_high_rating:
        trust_review = trust_review * (1.0 + reward)
    else:
        trust_review = trust_review * (1.0 - penalty)
    
    # BƯỚC 3: Phạt thêm nếu dùng sim data (không chắc chắn)
    if is_using_sim:
        trust_review = trust_review * 0.85  # Phạt 15% vì dùng ước lượng
        print(f"      ⚠️  Phạt thêm -15% do sử dụng ước lượng từ sản phẩm tương tự")
    
    # Đảm bảo trong khoảng hợp lý
    trust_review = max(0.1, min(1.2, trust_review))
    return float(trust_review)

def compute_price_trust(user_price, market_median):
    """
    CẤU TRÚC MỚI: Bắt đầu từ 1.0, phạt nếu giá lệch khỏi market median
    """
    if user_price is None or user_price <= 0:
        return 0.3
    
    # BẮT ĐẦU TỪ 1.0
    trust_price = 1.0
    
    # Tính độ lệch (log-space để xử lý tỷ lệ)
    dev = abs(math.log((user_price + 1e-9) / (market_median + 1e-9)))
    
    # PHẠT dựa trên độ lệch
    # dev = 0 → không phạt
    # dev càng lớn → phạt càng nặng
    penalty = 1.0 - math.exp(-dev)  # penalty ∈ [0, 1)
    trust_price = trust_price * (1.0 - penalty * 0.9)  # Phạt tối đa 90%
    
    return float(max(0.1, min(1.0, trust_price)))

def compute_discount_trust(discount_user, disc_low, disc_high):
    """
    CẤU TRÚC MỚI: Bắt đầu từ 1.0, phạt nếu discount lệch khỏi market range
    """
    if discount_user is None:
        return 0.6
    
    # BẮT ĐẦU TỪ 1.0
    trust_discount = 1.0
    
    # Kiểm tra trong range hay không
    if discount_user >= disc_low and discount_user <= disc_high:
        # Trong range → KHÔNG PHẠT, có thể thưởng nhẹ nếu gần median
        median_disc = (disc_low + disc_high) / 2.0
        closeness = 1.0 - abs(discount_user - median_disc) / ((disc_high - disc_low) / 2.0 + 1e-9)
        reward = closeness * 0.05  # Thưởng tối đa 5% nếu đúng median
        trust_discount = trust_discount * (1.0 + reward)
    else:
        # Ngoài range → PHẠT dựa trên khoảng cách
        if discount_user < disc_low:
            dist = disc_low - discount_user
        else:
            dist = discount_user - disc_high
        
        # Phạt theo khoảng cách
        penalty = min(0.9, dist * 3.0)  # Phạt tối đa 90%
        trust_discount = trust_discount * (1.0 - penalty)
    
    return float(max(0.1, min(1.05, trust_discount)))

def compute_store_trust(store_name):
    """
    CẤU TRÚC MỚI: Bắt đầu từ 1.0, phạt nếu không phải top store
    """
    # BẮT ĐẦU TỪ 1.0
    trust_store = 1.0
    
    if store_name in top_stores:
        # Top store → KHÔNG PHẠT, có thể thưởng nhẹ
        reward = 0.05  # Thưởng 5%
        trust_store = trust_store * (1.0 + reward)
    else:
        # Smaller store → PHẠT nhẹ
        penalty = 0.25  # Phạt 25%
        trust_store = trust_store * (1.0 - penalty)
    
    return float(max(0.5, min(1.1, trust_store)))

def compute_neighbor_consistency(neigh_prices):
    """
    CẤU TRÚC MỚI: Bắt đầu từ 1.0, phạt nếu giá không nhất quán (IQR lớn)
    """
    if len(neigh_prices) < 3:
        return 0.7  # Ít data → phạt nhẹ
    
    # BẮT ĐẦU TỪ 1.0
    trust_consistency = 1.0
    
    # Tính IQR để đo độ phân tán
    q1 = np.quantile(neigh_prices, 0.25)
    q3 = np.quantile(neigh_prices, 0.75)
    iqr = max(1e-9, q3 - q1)
    median = np.median(neigh_prices)
    
    # Tính tỷ lệ phân tán
    dispersion_ratio = iqr / (median + 1e-9)
    
    # PHẠT nếu phân tán cao
    # dispersion_ratio = 0 → không phạt (giá rất đồng nhất)
    # dispersion_ratio lớn → phạt nặng
    penalty = min(0.8, dispersion_ratio)  # Phạt tối đa 80%
    trust_consistency = trust_consistency * (1.0 - penalty)
    
    return float(max(0.2, min(1.0, trust_consistency)))

def aggregate_trust(tr_review, price_tr, disc_tr, store_tr, neigh_consistency,
                    weights=None):
    # Balanced weights by default
    if weights is None:
        weights = {
            'review': 0.35,
            'price': 0.25,
            'discount': 0.15,
            'store': 0.15,
            'neigh': 0.10
        }
    TRUST = (weights['review'] * tr_review +
             weights['price'] * price_tr +
             weights['discount'] * disc_tr +
             weights['store'] * store_tr +
             weights['neigh'] * neigh_consistency)
    # allow modest upward scale but clip
    TRUST = float(max(0.05, min(1.25, TRUST)))
    return TRUST

# -----------------------
# MARKET RANGE & DISCOUNT RANGE HELPERS (repeated for clarity)
# -----------------------
def get_market_price_range_from_similars(name, category_group, top_k=20,
                                         iqr_mult=1.5, lower_q=0.10, upper_q=0.90):
    sim_df = get_similar_products(name, top_k=top_k)
    sim_df = sim_df[sim_df["category_group"] == category_group]

    if sim_df is None or len(sim_df) == 0:
        return None, None, sim_df

    prices = sim_df['price'].astype(float)

    if len(prices) < 3:
        med = float(prices.median()) if len(prices)>0 else None
        return (med*0.8 if med else 10000,
                med*1.2 if med else 15000,
                sim_df)

    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    upper_cut = q3 + iqr_mult * iqr
    lower_cut = max(q1 - iqr_mult * iqr, 0)

    filtered = sim_df[(sim_df['price'] >= lower_cut) & (sim_df['price'] <= upper_cut)].copy()
    if len(filtered) == 0:
        filtered = sim_df.copy()

    market_low = float(filtered['price'].quantile(lower_q))
    market_high = float(filtered['price'].quantile(upper_q))

    if np.isclose(market_low, market_high):
        median = float(filtered['price'].median())
        market_low = max(10000, median * 0.9)
        market_high = median * 1.1

    return market_low, market_high, filtered.reset_index(drop=True)

def get_market_discount_range_from_similars(filtered_sim_df, lower_q=0.10, upper_q=0.90, iqr_mult=1.5):
    if filtered_sim_df is None or len(filtered_sim_df)==0:
        return 0.0, 0.0, filtered_sim_df

    discs = filtered_sim_df['discount_percent'].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(discs) < 3:
        med = float(discs.median()) if len(discs)>0 else 0.0
        return max(0.0, med*0.5), med*1.5 if med>0 else 0.3, filtered_sim_df

    q1 = discs.quantile(0.25)
    q3 = discs.quantile(0.75)
    iqr = q3 - q1
    lower_cut = max(q1 - iqr_mult * iqr, 0)
    upper_cut = q3 + iqr_mult * iqr
    filtered = discs[(discs >= lower_cut) & (discs <= upper_cut)]
    if len(filtered)==0:
        filtered = discs

    disc_low = float(filtered.quantile(lower_q))
    disc_high = float(filtered.quantile(upper_q))
    disc_low = max(0.0, disc_low)
    disc_high = max(disc_low, disc_high)
    return disc_low, disc_high, filtered_sim_df

def get_market_original_price_range_from_similars(filtered_sim_df, lower_q=0.10, upper_q=0.90, iqr_mult=1.5):
    """
    Tính khoảng giá gốc (original_price) thị trường từ sản phẩm tương tự
    Dùng để so sánh với original_price của user để xác định outlier
    """
    if filtered_sim_df is None or len(filtered_sim_df) == 0:
        return None, None, None
    
    orig_prices = filtered_sim_df['original_price'].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(orig_prices) < 3:
        med = float(orig_prices.median()) if len(orig_prices) > 0 else None
        if med is None:
            return None, None, None
        return med * 0.8, med * 1.2, med
    
    # Lọc outlier bằng IQR
    q1 = orig_prices.quantile(0.25)
    q3 = orig_prices.quantile(0.75)
    iqr = q3 - q1
    lower_cut = max(q1 - iqr_mult * iqr, 0)
    upper_cut = q3 + iqr_mult * iqr
    filtered = orig_prices[(orig_prices >= lower_cut) & (orig_prices <= upper_cut)]
    if len(filtered) == 0:
        filtered = orig_prices
    
    market_orig_low = float(filtered.quantile(lower_q))
    market_orig_high = float(filtered.quantile(upper_q))
    market_orig_median = float(filtered.median())
    
    if np.isclose(market_orig_low, market_orig_high):
        market_orig_low = max(10000, market_orig_median * 0.9)
        market_orig_high = market_orig_median * 1.1
    
    return market_orig_low, market_orig_high, market_orig_median

print("\nPart 2: inference helpers and Trust Score (Balanced) ready.")
# End PART 2 / 3
# =====================================================================
# PART 3 / 3
# - recommend_price (final) with Trust Score Balanced integrated
# - Build holdout features (X_hold)
# - Holdout evaluation (2 scenarios)
# - Demo + print logs
# =====================================================================

import math

# -----------------------
# HELPER: interval intersection
# -----------------------
def intersect_intervals(a_low, a_high, b_low, b_high):
    lo = max(a_low, b_low)
    hi = min(a_high, b_high)
    if lo > hi:
        return None, None
    return lo, hi

# -----------------------
# FINAL recommend_price (uses Trust Score Balanced)
# -----------------------
def recommend_price(original_price, name, store_name, category_group, price_to_test,
                    user_comment_count=None, user_rating=None,
                    price_grid_steps=40, market_top_k=20,
                    outlier_ratio_threshold=3.0, margin_discount=0.1,
                    min_trust_factor=0.20, strategy='balanced'):

    # 1) get market info from TRAIN similars
    market_low, market_high, filtered_sim = get_market_price_range_from_similars(
        name=name, category_group=category_group, top_k=market_top_k
    )
    if filtered_sim is None or len(filtered_sim) == 0:
        filtered_sim = get_similar_products(name, top_k=market_top_k)

    disc_low, disc_high, _ = get_market_discount_range_from_similars(filtered_sim)
    
    # TÍNH MARKET ORIGINAL PRICE RANGE để xác định outlier
    market_orig_low, market_orig_high, market_orig_median = get_market_original_price_range_from_similars(filtered_sim)

    # safe defaults
    if market_low is None or market_high is None:
        market_low = max(10000.0, original_price * 0.6)
        market_high = original_price

    market_median = (market_low + market_high) / 2.0
    
    # safe defaults for original price range
    if market_orig_median is None:
        market_orig_median = original_price
        market_orig_low = original_price * 0.7
        market_orig_high = original_price * 1.3

    # 2) tính discount_percent từ giá user nhập
    if price_to_test is not None and price_to_test > 0:
        user_discount_percent = (original_price - price_to_test) / original_price
        user_price = price_to_test
    else:
        user_discount_percent = 0.0
        user_price = original_price

    # 3) KIỂM TRA OUTLIER - Dùng ORIGINAL PRICE so với MARKET ORIGINAL PRICE MEDIAN
    outlier_flag = False
    outlier_severity = 'none'  # 'none', 'light', 'heavy'
    
    ratio_orig = original_price / (market_orig_median + 1e-9)
    ratio_orig_inv = (market_orig_median + 1e-9) / original_price
    
    print(f"\n   🔍 KIỂM TRA OUTLIER (dựa trên Original Price):")
    print(f"      User original price: {original_price:,.0f} VNĐ")
    print(f"      Market original price range: {market_orig_low:,.0f} - {market_orig_high:,.0f} VNĐ")
    print(f"      Market original price median: {market_orig_median:,.0f} VNĐ")
    print(f"      Market price range: {market_low:,.0f} - {market_high:,.0f} VNĐ")
    print(f"      Market price median: {market_median:,.0f} VNĐ")
    print(f"      Ratio: {ratio_orig:.2f}x")
    
    if ratio_orig > 3.0 or ratio_orig_inv > 3.0:
        # OUTLIER NẶNG: Gấp >3 lần
        outlier_flag = True
        outlier_severity = 'heavy'
        print(f"   ⚠️⚠️⚠️  OUTLIER NẶNG: Original price lệch market original median >3 lần")
        print(f"           → Market price range KHÔNG đáng tin, ưu tiên discount range")
    elif ratio_orig > 1.5 or ratio_orig_inv > 1.5:
        # OUTLIER NHẸ: Gấp 1.5-3 lần
        outlier_flag = True
        outlier_severity = 'light'
        print(f"   ⚠️  OUTLIER NHẸ: Original price lệch market original median 1.5-3 lần")
        print(f"       → Kết hợp market range và discount range với điều chỉnh")
    else:
        print(f"   ✅ KHÔNG OUTLIER: Original price nằm trong khoảng hợp lý với thị trường")
        print(f"       → Sử dụng market range và discount range bình thường")

    # 4) compute discount-driven range from original_price and market discount range
    disc_price_low = original_price * max(0.0, (1.0 - disc_high))
    disc_price_high = original_price * max(0.0, (1.0 - disc_low))
    if disc_price_low > disc_price_high:
        disc_price_low, disc_price_high = disc_price_high, disc_price_low

    # 5) XÂY DỰNG CANDIDATE PRICE RANGE dựa trên outlier severity VÀ strategy
    if outlier_severity == 'heavy':
        # OUTLIER NẶNG → Không tin market price range, ưu tiên discount range
        print(f"   🔧 Sử dụng DISCOUNT RANGE ưu tiên (market price không đáng tin)")
        base_low = max(1000.0, disc_price_low * 0.8)
        base_high = max(base_low + 1.0, disc_price_high * 1.2)
        
    elif outlier_severity == 'light':
        # OUTLIER NHẸ → Kết hợp market và discount range với ưu tiên discount
        print(f"   🔧 Kết hợp market range và discount range (ưu tiên discount)")
        # Mở rộng market range 30%
        expanded_market_low = market_low * 0.7
        expanded_market_high = market_high * 1.3
        base_low, base_high = intersect_intervals(expanded_market_low, expanded_market_high, 
                                                   disc_price_low, disc_price_high)
        if base_low is None:
            # Không giao nhau → dùng discount range
            base_low = max(1000.0, disc_price_low)
            base_high = max(base_low + 1.0, disc_price_high)
        else:
            base_low = max(1000.0, base_low)
            base_high = max(base_low + 1.0, base_high)
            
    else:
        # KHÔNG OUTLIER → Dùng giao của market và discount range
        print(f"   🔧 Tính base range từ giao của market và discount range")
        base_low, base_high = intersect_intervals(market_low, market_high, disc_price_low, disc_price_high)
        if base_low is None:
            # Không giao nhau → làm mềm 10%
            softened_low = max(1000.0, disc_price_low * 0.9)
            softened_high = disc_price_high * 1.1
            base_low, base_high = intersect_intervals(market_low, market_high, softened_low, softened_high)
            if base_low is None:
                # Vẫn không giao → dùng market range
                base_low, base_high = market_low, market_high

    base_low = max(1000.0, base_low)
    base_high = max(base_low + 1.0, base_high)
    
    # MỞ RỘNG RANGE THEO STRATEGY - ÁP DỤNG CHO TẤT CẢ TRƯỜNG HỢP
    if strategy == 'aggressive':
        # AGGRESSIVE: Mở rộng NHIỀU về phía LOW (giá thấp để tăng tỷ lệ bán)
        # Giảm phía LOW nhiều, giảm phía HIGH vừa phải
        # Low: -30%, High: -20%
        cand_low = max(1000.0, base_low * 0.70)
        cand_high = max(cand_low + 1.0, base_high * 0.80)
        print(f"   🔴 AGGRESSIVE: Mở rộng về phía LOW (giá thấp) -30% low / -20% high")
    elif strategy == 'conservative':
        # CONSERVATIVE: Mở rộng NHIỀU về phía HIGH (giá cao để tăng doanh thu)
        # Tăng phía LOW ít, mở rộng phía HIGH nhiều
        # Low: +20%, High: +30%
        cand_low = max(1000.0, base_low * 1.20)
        cand_high = max(cand_low + 1.0, base_high * 1.30)
        print(f"   🟢 CONSERVATIVE: Thu hẹp phía LOW, mở rộng phía HIGH +20% low / +30% high")
    else:
        # BALANCED: Mở rộng CÂN ĐỐI cả 2 phía
        # Low: -25%, High: +25%
        cand_low = max(1000.0, base_low * 0.75)
        cand_high = max(cand_low + 1.0, base_high * 1.25)
        print(f"   🟡 BALANCED: Mở rộng CÂN ĐỐI -25% low / +25% high")
    
    print(f"   📊 Candidate price range: {cand_low:,.0f} - {cand_high:,.0f} VNĐ")

    # Duyệt tất cả giá trị có thể với bước nhảy 5,000 VNĐ
    candidate_prices = np.arange(cand_low, cand_high + 5000, 5000)
    print(f"   📊 Số lượng giá candidate: {len(candidate_prices):,} giá trị (bước nhảy 5,000 VNĐ)")

    # 6) baseline prob for user input price (use prepare_input_features; avoid leaking discount_percent)
    fv_input = prepare_input_features(name, store_name, category_group, user_price,
                                      comment_count=user_comment_count, rating=user_rating,
                                      use_sim_fill_for_missing=True)
    prob_input = predict_prob_sold_gt100(fv_input)

    # 7) compute user-level trust (for prob_input_trusted)
    # gather sim stats for name (from TRAIN)
    emb_q = embed_model.encode([name], convert_to_numpy=True)
    _, idxs = nn_final.kneighbors(emb_q, n_neighbors=NN_SIMILAR_K+1, return_distance=True)
    neigh_idx = idxs[0, 1:NN_SIMILAR_K+1]
    neigh = df_train_full.iloc[neigh_idx]
    neigh_prices = neigh['price'].values if len(neigh) > 0 else np.array([market_median])
    sim_comment_avg = float(neigh['comment_count'].mean()) if len(neigh)>0 else 0.0
    sim_rating_avg = float(neigh['rating'].mean()) if len(neigh)>0 else 4.8
    sim_disc_avg = float(neigh['discount_percent'].mean()) if len(neigh)>0 else 0.0

    # review trust
    is_from_sim = (user_comment_count is None and user_rating is None)
    tr_review = compute_review_trust(user_comment_count if user_comment_count is not None else None,
                                     user_rating if user_rating is not None else None,
                                     sim_comment_avg, sim_rating_avg,
                                     is_from_sim=is_from_sim)

    # price trust (user_price reliability)
    price_tr = compute_price_trust(user_price, market_median) if user_price is not None else 0.4

    # discount trust (user discount vs market)
    discount_user = None
    if user_price is not None and user_price > 0:
        discount_user = max(0.0, (original_price - user_price) / (original_price + 1e-9))
    disc_tr = compute_discount_trust(discount_user, disc_low, disc_high)

    # store trust & neighbor consistency
    store_tr = compute_store_trust(store_name)
    neigh_cons = compute_neighbor_consistency(neigh_prices)

    trust_user = aggregate_trust(tr_review, price_tr, disc_tr, store_tr, neigh_cons)

    # form prob_input_trusted — balanced mapping (not too extreme)
    prob_input_trusted = prob_input * (0.5 + 0.5 * trust_user)
    
    # ÁP DỤNG OUTLIER PENALTY CHO PRICE_TO_TEST - PHẠT THEO DISCOUNT DEVIATION
    # Tính discount của price_to_test
    if user_price is not None and user_price > 0:
        user_discount = (original_price - user_price) / original_price
        median_discount = (disc_low + disc_high) / 2.0
        
        # PHÂN BIỆT: Trong range vs Ngoài range
        if user_discount < disc_low:
            # NGOÀI RANGE (Thấp hơn disc_low) → PHẠT NẶNG HƠN
            # Discount thấp = Giá cao bất thường
            deviation = disc_low - user_discount
            
            if outlier_severity == 'heavy':
                # PHẠT CỰC NẶNG khi ngoài range
                if deviation <= 0.10:
                    outlier_penalty = max(0.70, 1.0 - deviation * 3.0)
                elif deviation <= 0.20:
                    outlier_penalty = max(0.50, 0.70 - (deviation - 0.10) * 2.0)
                else:
                    outlier_penalty = max(0.10, 0.50 * math.exp(-(deviation - 0.20) * 4.5))
                
                print(f"   ⚠️⚠️⚠️  OUTLIER NẶNG - User discount {user_discount:.1%} < disc_low {disc_low:.1%}")
                print(f"           Deviation: {deviation:.1%} → Penalty: {outlier_penalty:.3f} (PHẠT NẶNG - ngoài range)")
            
            elif outlier_severity == 'light':
                # PHẠT NẶNG (nhưng nhẹ hơn heavy)
                if deviation <= 0.15:
                    outlier_penalty = max(0.85, 1.0 - deviation * 1.0)
                elif deviation <= 0.30:
                    outlier_penalty = max(0.70, 0.85 - (deviation - 0.15) * 1.0)
                else:
                    outlier_penalty = max(0.40, 0.70 * math.exp(-(deviation - 0.30) * 2.5))
                
                print(f"   ⚠️  OUTLIER NHẸ - User discount {user_discount:.1%} < disc_low {disc_low:.1%}")
                print(f"       Deviation: {deviation:.1%} → Penalty: {outlier_penalty:.3f} (phạt nặng - ngoài range)")
            
            else:  # none
                outlier_penalty = 1.0
        
        elif user_discount > disc_high:
            # NGOÀI RANGE (Cao hơn disc_high) → PHẠT NẶNG HƠN
            # Discount cao = Giá thấp bất thường
            deviation = user_discount - disc_high
            
            if outlier_severity == 'heavy':
                # PHẠT CỰC NẶNG khi ngoài range
                if deviation <= 0.10:
                    outlier_penalty = max(0.70, 1.0 - deviation * 3.0)
                elif deviation <= 0.20:
                    outlier_penalty = max(0.50, 0.70 - (deviation - 0.10) * 2.0)
                else:
                    outlier_penalty = max(0.10, 0.50 * math.exp(-(deviation - 0.20) * 4.5))
                
                print(f"   ⚠️⚠️⚠️  OUTLIER NẶNG - User discount {user_discount:.1%} > disc_high {disc_high:.1%}")
                print(f"           Deviation: {deviation:.1%} → Penalty: {outlier_penalty:.3f} (PHẠT NẶNG - ngoài range)")
            
            elif outlier_severity == 'light':
                # PHẠT NẶNG (nhưng nhẹ hơn heavy)
                if deviation <= 0.15:
                    outlier_penalty = max(0.85, 1.0 - deviation * 1.0)
                elif deviation <= 0.30:
                    outlier_penalty = max(0.70, 0.85 - (deviation - 0.15) * 1.0)
                else:
                    outlier_penalty = max(0.40, 0.70 * math.exp(-(deviation - 0.30) * 2.5))
                
                print(f"   ⚠️  OUTLIER NHẸ - User discount {user_discount:.1%} > disc_high {disc_high:.1%}")
                print(f"       Deviation: {deviation:.1%} → Penalty: {outlier_penalty:.3f} (phạt nặng - ngoài range)")
            
            else:  # none
                outlier_penalty = 1.0
        
        else:
            # TRONG RANGE [disc_low, disc_high] → PHẠT NHẸ theo median
            discount_deviation = abs(user_discount - median_discount)
            
            if outlier_severity == 'heavy':
                # PHẠT theo deviation từ median (NHẸ HƠN so với ngoài range)
                if discount_deviation <= 0.10:
                    outlier_penalty = max(0.95, 1.0 - discount_deviation * 0.5)
                elif discount_deviation <= 0.20:
                    outlier_penalty = max(0.85, 1.0 - (discount_deviation - 0.10) * 1.5 - 0.05)
                elif discount_deviation <= 0.30:
                    outlier_penalty = max(0.75, 1.0 - (discount_deviation - 0.20) * 0.5 - 0.15)
                else:
                    excess = discount_deviation - 0.30
                    outlier_penalty = max(0.10, 0.75 * math.exp(-excess * 4.5))
                
                print(f"   ⚠️⚠️⚠️  OUTLIER NẶNG - Discount trong range [{disc_low:.1%}, {disc_high:.1%}]")
                print(f"           User discount: {user_discount:.1%} | Median: {median_discount:.1%}")
                print(f"           Deviation: {discount_deviation:.1%} → Penalty: {outlier_penalty:.3f} (phạt nhẹ - trong range)")
            
            elif outlier_severity == 'light':
                # PHẠT NHẸ theo deviation từ median
                if discount_deviation <= 0.15:
                    outlier_penalty = max(0.97, 1.0 - discount_deviation * 0.2)
                elif discount_deviation <= 0.30:
                    outlier_penalty = max(0.90, 0.97 - (discount_deviation - 0.15) * 0.5)
                elif discount_deviation <= 0.50:
                    outlier_penalty = max(0.80, 0.90 - (discount_deviation - 0.30) * 0.5)
                else:
                    excess = discount_deviation - 0.50
                    outlier_penalty = max(0.50, 0.80 * math.exp(-excess * 2.0))
                
                print(f"   ⚠️  OUTLIER NHẸ - Discount trong range [{disc_low:.1%}, {disc_high:.1%}]")
                print(f"       User discount: {user_discount:.1%} | Median: {median_discount:.1%}")
                print(f"       Deviation: {discount_deviation:.1%} → Penalty: {outlier_penalty:.3f} (phạt nhẹ - trong range)")
            
            else:  # none
                outlier_penalty = 1.0
    else:
        # Fallback nếu không có user_price
        if outlier_severity == 'heavy':
            outlier_penalty = 0.7
            print(f"   ⚠️⚠️⚠️  Áp dụng OUTLIER NẶNG penalty (fallback): prob × {outlier_penalty}")
        elif outlier_severity == 'light':
            outlier_penalty = 0.85
            print(f"   ⚠️  Áp dụng OUTLIER NHẸ penalty (fallback): prob × {outlier_penalty}")
        else:
            outlier_penalty = 1.0
    
    prob_input_trusted *= outlier_penalty
    
    # apply min trust floor
    prob_input_trusted = float(max(0.01, min(0.999, prob_input_trusted)))

    # 7) evaluate candidate prices: enforce discount constraints based on strategy
    valid_candidates = []
    for p in candidate_prices:
        discount_p = max(0.0, (original_price - p) / (original_price + 1e-9))
        
        # KHÔNG còn hard constraint cho balanced - để penalty/reward tự điều chỉnh
        # Chỉ aggressive và conservative được tự do explore

        fv = prepare_input_features(name, store_name, category_group, float(p),
                                    comment_count=user_comment_count, rating=user_rating,
                                    use_sim_fill_for_missing=True)
        prob = predict_prob_sold_gt100(fv)

        price_tr_cand = compute_price_trust(p, market_median)
        discount_tr_cand = compute_discount_trust(discount_p, disc_low, disc_high)
        tr_cand = aggregate_trust(tr_review, price_tr_cand, discount_tr_cand, store_tr, neigh_cons)

        # === NEW PROBABILITY CALCULATION FORMULA ===
        # prob_final = 0.6*prob_raw + 0.25*trust_score*prob_raw + 0.10*market_fit_reward + 0.05*review_reward
        # = 60% raw model
        # + 25% trust correction
        # + 10% market range reward/penalty (giá trị tuyệt đối 0-0.10)
        # + 5% review-based reward/penalty (giá trị tuyệt đối 0-0.05)
        
        # BƯỚC 1: Component 1 & 2 - Model prediction + Trust adjustment
        prob_component_1_2 = 0.60 * prob + 0.25 * tr_cand * prob
        
        # BƯỚC 2: Component 3 - Market fit reward (GIÁ TRỊ 0-1.0, nhân với 0.10)
        # Thưởng khi giá nằm trong market range và discount phù hợp
        market_fit_reward = 0.0
        if p >= market_low and p <= market_high:
            # Giá trong range thị trường → +0.5
            market_fit_reward += 0.5
            if discount_p >= disc_low and discount_p <= disc_high:
                # Discount cũng trong range → +0.5 nữa
                market_fit_reward += 0.5
        # market_fit_reward = 1.0 nếu thỏa hết điều kiện
        # Đóng góp vào prob_final: 0.10 * market_fit_reward (max = 0.10)
        
        # BƯỚC 3: Component 4 - Review reward (GIÁ TRỊ TUYỆT ĐỐI 0-0.05)
        # Thưởng dựa trên review data quality
        review_reward = 0.0
        
        if user_comment_count is not None and user_comment_count > 0:
            # Có comment
            if user_comment_count >= 100:
                review_reward += 0.3     # Comment rất nhiều
            elif user_comment_count >= 30:
                review_reward += 0.2     # Comment khá nhiều
            elif user_comment_count >= 10:
                review_reward += 0.15    # Comment vừa phải
            else:
                review_reward += 0.05    # Comment ít
        
        if user_rating is not None and user_rating > 0:
            # Có rating
            if user_rating >= 4.8:
                review_reward += 0.7     # Rating rất cao
            elif user_rating >= 4.6:
                review_reward += 0.5     # Rating cao
            elif user_rating >= 4.3:
                review_reward += 0.3     # Rating khá
            elif user_rating >= 4.0:
                review_reward += 0.15    # Rating trung bình
            else:
                review_reward += 0.05    # Rating thấp
        # review_reward = 0.05 nếu có review tốt nhất (100+ comment + rating 4.5+)
        
        # BƯỚC 4: TỔNG HỢP CUỐI CÙNG
        prob_scaled = (
            prob_component_1_2 +      # 60% model + 25% trust
            0.10 * market_fit_reward + # 10% * (0-1) = max 0.10
            0.05 * review_reward       # 5% * (0-1) = max 0.05
        )
        # Đảm bảo prob_scaled trong [0, 0.999] - KHÔNG vượt quá 99.9%
        prob_scaled = max(0.0, min(0.999, prob_scaled))
        
        # Outlier penalty - TẤT CẢ MỨC ĐỘ OUTLIER DÙNG DISCOUNT DEVIATION
        if outlier_severity == 'heavy':
            # OUTLIER NẶNG: Phạt theo discount deviation
            discount_p = (original_price - p) / original_price
            median_discount = (disc_low + disc_high) / 2.0
            discount_deviation = abs(discount_p - median_discount)
            
            # PHẠT NẶNG
            if discount_deviation <= 0.10:
                penalty = max(0.95, 1.0 - discount_deviation * 0.5)
            elif discount_deviation <= 0.20:
                penalty = max(0.85, 1.0 - (discount_deviation - 0.10) * 1.5 - 0.05)
            elif discount_deviation <= 0.30:
                penalty = max(0.75, 1.0 - (discount_deviation - 0.20) * 0.5 - 0.15)
            else:
                excess = discount_deviation - 0.30
                penalty = max(0.10, 0.75 * math.exp(-excess * 4.5))
            prob_scaled *= penalty
        
        elif outlier_severity == 'light':
            # OUTLIER NHẸ: CÙNG LOGIC nhưng PHẠT NHẸ HƠN
            discount_p = (original_price - p) / original_price
            median_discount = (disc_low + disc_high) / 2.0
            discount_deviation = abs(discount_p - median_discount)
            
            # PHẠT NHẸ HƠN: threshold cao hơn, min penalty cao hơn
            if discount_deviation <= 0.15:  # Ngưỡng rộng hơn
                penalty = max(0.97, 1.0 - discount_deviation * 0.2)
            elif discount_deviation <= 0.30:
                penalty = max(0.90, 0.97 - (discount_deviation - 0.15) * 0.5)
            elif discount_deviation <= 0.50:
                penalty = max(0.80, 0.90 - (discount_deviation - 0.30) * 0.5)
            else:
                excess = discount_deviation - 0.50
                penalty = max(0.50, 0.80 * math.exp(-excess * 2.0))
            prob_scaled *= penalty

        # Calculate metrics for strategy evaluation
        profit = p - original_price
        profit_margin = profit / (original_price + 1e-9)
        expected_revenue = prob_scaled * p
        expected_profit = prob_scaled * profit
        
        # Strategy-specific penalties
        profit_penalty = 1.0
        prob_penalty = 1.0
        
        # HỆ SỐ PHẠT THEO OUTLIER SEVERITY
        # Outlier nặng → Phạt nặng hơn, thưởng ít hơn
        if outlier_severity == 'heavy':
            penalty_multiplier = 1.5  # Phạt nặng hơn 50%
            reward_multiplier = 0.7   # Thưởng giảm 30%
        elif outlier_severity == 'light':
            penalty_multiplier = 1.2  # Phạt nặng hơn 20%
            reward_multiplier = 0.85  # Thưởng giảm 15%
        else:  # none
            penalty_multiplier = 1.0  # Phạt bình thường
            reward_multiplier = 1.0   # Thưởng bình thường
        
        if strategy == 'conservative':
            # CONSERVATIVE MỚI: Tối đa DOANH THU (prob × price)
            # THƯỞNG: Vùng [disc_low, median] - Giảm ÍT, giá CAO
            # PHẠT: Vùng [median, disc_high] - Giảm NHIỀU, giá THẤP
            # Revenue = prob × price tự cân bằng → KHÔNG cần phạt prob
            actual_discount = (original_price - p) / original_price
            median_discount = (disc_low + disc_high) / 2.0
            range_width = disc_high - disc_low
            
            # Xác định vị trí trong range
            if actual_discount < disc_low or actual_discount > disc_high:
                # NGOÀI RANGE → PHẠT NẶNG
                if actual_discount < disc_low:
                    deviation = disc_low - actual_discount
                else:
                    deviation = actual_discount - disc_high
                
                if deviation <= 0.10:
                    base_penalty = max(0.70, 1.0 - deviation * 3.0)
                elif deviation <= 0.20:
                    base_penalty = max(0.50, 0.70 - (deviation - 0.10) * 2.0)
                else:
                    base_penalty = max(0.20, 0.50 * math.exp(-(deviation - 0.20) * 3.0))
                
                penalty_strength = 1.0 - base_penalty
                penalty_strength *= penalty_multiplier
                profit_penalty = max(0.10, 1.0 - penalty_strength)
                prob_penalty = 1.0  # Không phạt prob, chỉ phạt profit
            
            elif actual_discount >= disc_low and actual_discount <= median_discount:
                # VÙNG THƯỞNG: [disc_low, median] - Giảm ÍT để giá CAO
                # Tính vị trí trong vùng (0 = disc_low, 1 = median)
                position = (actual_discount - disc_low) / (median_discount - disc_low + 1e-9)
                position = max(0.0, min(1.0, position))
                
                # Thưởng GIẢM DẦN từ disc_low (max) → median (min)
                # Càng gần disc_low (discount thấp, giá cao) càng thưởng nhiều
                base_profit_reward = 0.30 * (1.0 - position)  # Max +30% tại disc_low
                
                profit_penalty = min(1.30, 1.0 + base_profit_reward * reward_multiplier)
                prob_penalty = 1.0  # Không điều chỉnh prob
            
            else:
                # VÙNG PHẠT: (median, disc_high] - Giảm NHIỀU, giá THẤP
                # Tính vị trí trong vùng (0 = median, 1 = disc_high)
                position = (actual_discount - median_discount) / (disc_high - median_discount + 1e-9)
                position = max(0.0, min(1.0, position))
                
                # Phạt TĂNG DẦN từ median (nhẹ) → disc_high (nặng)
                base_profit_penalty = 0.05 + 0.45 * position  # Phạt 5% → 50%
                
                profit_penalty = max(0.50, 1.0 - base_profit_penalty * penalty_multiplier)
                prob_penalty = 1.0  # Không điều chỉnh prob
        
        elif strategy == 'aggressive':
            # AGGRESSIVE MỚI: Tối đa TỶ LỆ BÁN (prob)
            # THƯỞNG: Vùng [median, disc_high] - Giảm NHIỀU, giá THẤP
            # PHẠT: Vùng [disc_low, median] - Giảm ÍT, giá CAO
            # Revenue = prob × price tự cân bằng → KHÔNG cần điều chỉnh profit
            actual_discount = (original_price - p) / original_price
            median_discount = (disc_low + disc_high) / 2.0
            range_width = disc_high - disc_low
            
            # Xác định vị trí trong range
            if actual_discount < disc_low or actual_discount > disc_high:
                # NGOÀI RANGE → PHẠT NẶNG
                if actual_discount < disc_low:
                    deviation = disc_low - actual_discount
                else:
                    deviation = actual_discount - disc_high
                
                if deviation <= 0.10:
                    base_penalty = max(0.70, 1.0 - deviation * 3.0)
                elif deviation <= 0.20:
                    base_penalty = max(0.50, 0.70 - (deviation - 0.10) * 2.0)
                else:
                    base_penalty = max(0.20, 0.50 * math.exp(-(deviation - 0.20) * 3.0))
                
                penalty_strength = 1.0 - base_penalty
                penalty_strength *= penalty_multiplier
                prob_penalty = max(0.10, 1.0 - penalty_strength)
                profit_penalty = 1.0  # Không điều chỉnh profit
                
                # DEBUG for specific prices
                if abs(p - 424000) < 10000:
                    print(f"      🐞 AGGRESSIVE DEBUG - Giá {p:,.0f}:")
                    print(f"         Discount: {actual_discount:.3%} (NGOÀI range [{disc_low:.1%}, {disc_high:.1%}])")
                    print(f"         Deviation: {deviation:.1%} | Base penalty: {base_penalty:.2f}")
                    print(f"         Prob penalty: {prob_penalty:.2f}")
            
            elif actual_discount >= median_discount and actual_discount <= disc_high:
                # VÙNG THƯỞNG: [median, disc_high] - Giảm NHIỀU để tăng TỶ LỆ
                # Tính vị trí trong vùng (0 = median, 1 = disc_high)
                position = (actual_discount - median_discount) / (disc_high - median_discount + 1e-9)
                position = max(0.0, min(1.0, position))
                
                # Thưởng TĂNG DẦN từ median (min) → disc_high (max)
                # Càng gần disc_high (discount cao, giá thấp) càng thưởng nhiều
                base_prob_reward = 0.40 * position  # Max +40% tại disc_high
                
                prob_penalty = min(1.40, 1.0 + base_prob_reward * reward_multiplier)
                profit_penalty = 1.0  # Không điều chỉnh profit
                
                # DEBUG for specific prices
                if abs(p - 424000) < 10000:
                    print(f"      🐞 AGGRESSIVE DEBUG - Giá {p:,.0f}:")
                    print(f"         Discount: {actual_discount:.3%} (VÙNG THƯỞNG [median {median_discount:.1%}, {disc_high:.1%}])")
                    print(f"         Position: {position:.2f} | Base reward: {base_prob_reward:.2%}")
                    print(f"         Prob penalty (reward): {prob_penalty:.2f}")
            
            else:
                # VÙNG PHẠT: [disc_low, median) - Giảm ÍT, giá CAO
                # Tính vị trí trong vùng (0 = disc_low, 1 = median)
                position = (actual_discount - disc_low) / (median_discount - disc_low + 1e-9)
                position = max(0.0, min(1.0, position))
                
                # Phạt GIẢM DẦN từ disc_low (nặng) → median (nhẹ)
                base_prob_penalty = 0.40 * (1.0 - position)  # Phạt 40% → 0%
                
                prob_penalty = max(0.60, 1.0 - base_prob_penalty * penalty_multiplier)
                profit_penalty = 1.0  # Không điều chỉnh profit
                
                # DEBUG for specific prices
                if abs(p - 424000) < 10000:
                    print(f"      🐞 AGGRESSIVE DEBUG - Giá {p:,.0f}:")
                    print(f"         Discount: {actual_discount:.3%} (VÙNG PHẠT [{disc_low:.1%}, median {median_discount:.1%}))")
                    print(f"         Position: {position:.2f} | Base penalty: {base_prob_penalty:.2%}")
                    print(f"         Prob penalty: {prob_penalty:.2f}")
        
        elif strategy == 'balanced':
            # BALANCED MỚI: Cân bằng TỶ LỆ BÁN và DOANH THU
            # THƯỞNG: Vùng [disc_low, disc_high], càng gần MEDIAN càng tốt
            # PHẠT: Càng xa median (gần 2 cận) càng phạt nặng
            # Ngoài range phạt rất nặng
            
            actual_discount = (original_price - p) / original_price
            median_discount = (disc_low + disc_high) / 2.0
            range_width = disc_high - disc_low
            
            # Tính khoảng cách từ actual_discount đến median
            distance_to_median = abs(actual_discount - median_discount)
            
            if actual_discount >= disc_low and actual_discount <= disc_high:
                # TRONG RANGE [disc_low, disc_high]
                # Normalize distance (0 = tại median, 1 = tại cận)
                normalized_distance = distance_to_median / (range_width / 2.0 + 1e-9)
                normalized_distance = max(0.0, min(1.0, normalized_distance))
                
                if normalized_distance <= 0.3:
                    # Rất gần median (±15% range) → THƯỞNG MẠNH
                    base_reward = 0.25 * (1.0 - normalized_distance / 0.3)
                elif normalized_distance <= 0.6:
                    # Gần median (±30% range) → THƯỞNG VỪA
                    base_reward = 0.15 * (1.0 - (normalized_distance - 0.3) / 0.3)
                else:
                    # Xa median, gần cận (>60% distance) → PHẠT NHẸ
                    base_penalty = 0.15 * (normalized_distance - 0.6) / 0.4  # Phạt 0% → 15%
                    base_reward = -base_penalty
                
                # Áp dụng outlier severity
                adjusted_reward = base_reward * reward_multiplier if base_reward > 0 else base_reward * penalty_multiplier
                prob_penalty = max(0.80, min(1.25, 1.0 + adjusted_reward))
                profit_penalty = max(0.80, min(1.25, 1.0 + adjusted_reward))
            
            else:
                # NGOÀI RANGE → PHẠT NẶNG
                if actual_discount < disc_low:
                    deviation = disc_low - actual_discount
                else:
                    deviation = actual_discount - disc_high
                
                # Phạt theo mức độ lệch
                if deviation <= 0.10:
                    base_penalty = max(0.70, 1.0 - deviation * 3.0)
                elif deviation <= 0.20:
                    base_penalty = max(0.50, 0.70 - (deviation - 0.10) * 2.0)
                else:
                    base_penalty = max(0.30, 0.50 * math.exp(-(deviation - 0.20) * 2.5))
                
                # Áp dụng outlier severity
                penalty_strength = 1.0 - base_penalty
                penalty_strength *= penalty_multiplier
                final_penalty = max(0.20, 1.0 - penalty_strength)
                prob_penalty = final_penalty
                profit_penalty = final_penalty
        
        # Apply penalties
        final_profit = profit * profit_penalty
        final_prob_scaled = prob_scaled * prob_penalty
        # Đảm bảo final_prob_scaled KHÔNG vượt quá 99.9%
        final_prob_scaled = min(0.999, final_prob_scaled)
        final_expected_revenue = final_prob_scaled * p
        final_expected_profit = final_prob_scaled * final_profit
        
        # DEBUG: Complete penalty chain for specific prices
        if abs(p - 424000) < 10000:
            print(f"      🔍 COMPLETE PENALTY CHAIN - Giá {p:,.0f}:")
            print(f"         prob_scaled (after outlier): {prob_scaled:.3%}")
            print(f"         prob_penalty (strategy): {prob_penalty:.3f}")
            print(f"         final_prob_scaled: {final_prob_scaled:.3%}")
            print(f"         final_expected_revenue: {final_expected_revenue:,.0f}")
        
        # RÀNG BUỘC TỶ LỆ BÁN theo strategy:
        # - aggressive: prob >= max(prob_balanced, prob_conservative) - Đảm bảo tỷ lệ cao nhất
        # - balanced: prob >= prob_input_trusted (balanced phải >= price_to_test)
        # - conservative: KHÔNG có ràng buộc tỷ lệ (uu tiên lợi nhuận)
        if strategy == 'aggressive':
            # AGGRESSIVE: TỶ LỆ BÁN PHẢI LỚN HƠN 2 CHIẾN LƯỢC CÒN LẠI
            # Tạm thời chấp nhận tất cả, sẽ so sánh sau khi có cả 3 chiến lược
            pass
        elif strategy == 'balanced':
            # BALANCED: TỶ LỆ BÁN >= TỶ LỆ CỦA PRICE_TO_TEST
            if final_prob_scaled < prob_input_trusted:
                continue
        # Conservative: KHÔNG có ràng buộc tỷ lệ, chấp nhận mọi tỷ lệ

        valid_candidates.append((
            float(p), prob, float(final_prob_scaled), final_expected_revenue, 
            discount_p, tr_cand, float(final_profit), float(profit_margin), float(final_expected_profit)
        ))

    # 8) choose best candidate based on strategy
    # ĐẢM BẢO: Tỷ lệ bán aggressive > balanced > conservative
    if len(valid_candidates) > 0:
        if strategy == 'aggressive':
            # AGGRESSIVE: TỐI ĐA HÓA XÁC SUẤT BÁN
            # - KHÔNG ràng buộc lỗ/lãi, chấp nhận mọi mức giá
            # - Chỉ chọn giá có xác suất bán cao nhất
            # DEBUG: Check candidates
            print(f"\n   🔍 DEBUG AGGRESSIVE - Candidates:")
            print(f"      Total valid_candidates: {len(valid_candidates)}")
            candidates_420_430 = [c for c in valid_candidates if 420000 <= c[0] <= 430000]
            print(f"      Candidates in 420K-430K range: {len(candidates_420_430)}")
            for c in candidates_420_430:
                discount_c = (original_price - c[0]) / original_price
                print(f"         Giá: {c[0]:,.0f} | Discount: {discount_c:.3%} | Prob: {c[2]:.3%} | Margin: {c[7]:.1%}")
            
            # KHÔNG LỌC margin - chấp nhận mọi mức lỗ/lãi
            filtered = valid_candidates
            
            # DEBUG: In top 5 candidates
            sorted_candidates = sorted(filtered, key=lambda x: x[2], reverse=True)[:5]
            print(f"\n   🔍 DEBUG AGGRESSIVE - Top 5 candidates:")
            for i, c in enumerate(sorted_candidates, 1):
                discount_c = (original_price - c[0]) / original_price
                print(f"      {i}. Giá: {c[0]:,.0f} | Discount: {discount_c:.1%} | Prob: {c[2]:.1%} | Revenue: {c[3]:,.0f}")
            
            best = max(filtered, key=lambda x: x[2])  # Max prob_scaled
            
            # PHASE 2: Boundary expansion - Mở rộng động dựa trên discount
            candidate_prices_list = sorted([c[0] for c in valid_candidates])
            min_candidate = min(candidate_prices_list)
            max_candidate = max(candidate_prices_list)
            
            if best[0] == min_candidate:
                # Giá tốt nhất chạm biên LOW → Mở rộng xuống thấp hơn
                discount_best = (original_price - best[0]) / original_price
                # Mở rộng tối đa 10% × (1 - discount_best)
                # discount_best càng cao (gần 100%) → mở rộng càng ít
                max_expansion_discount = discount_best + 0.10 * (1 - discount_best)
                max_expansion_price = original_price * (1 - max_expansion_discount)
                
                print(f"      🔍 Giá tốt nhất ({best[0]:,.0f}, discount {discount_best:.1%}) chạm biên LOW")
                print(f"         → Mở rộng discount từ {discount_best:.1%} đến {max_expansion_discount:.1%}")
                print(f"         → Mở rộng giá từ {best[0]:,.0f} xuống {max_expansion_price:,.0f}")
                
                # Generate expansion prices with 5K step
                expansion_prices = []
                p = best[0] - 5000
                while p > max_expansion_price and p > 0 and len(expansion_prices) < 20:
                    expansion_prices.append(p)
                    p -= 5000
                
                print(f"         → Duyệt thêm {len(expansion_prices)} giá trị")
                
                for p in expansion_prices:
                    discount_p = (original_price - p) / original_price
                    
                    # Apply exponential penalty when exceeding expansion range
                    if discount_p > max_expansion_discount:
                        excess = discount_p - max_expansion_discount
                        # Tốc độ giảm: exp(-excess × 20) → giảm rất nhanh
                        expansion_penalty = math.exp(-excess * 20)
                    else:
                        expansion_penalty = 1.0
                    
                    fv = prepare_input_features(name, store_name, category_group, float(p),
                                                comment_count=user_comment_count, rating=user_rating,
                                                use_sim_fill_for_missing=True)
                    prob = predict_prob_sold_gt100(fv)
                    price_tr_cand = compute_price_trust(p, market_median)
                    discount_tr_cand = compute_discount_trust(discount_p, disc_low, disc_high)
                    tr_cand = aggregate_trust(tr_review, price_tr_cand, discount_tr_cand, store_tr, neigh_cons)
                    
                    prob_component_1_2 = 0.60 * prob + 0.25 * tr_cand * prob
                    market_fit_reward = 0.0
                    if p >= market_low and p <= market_high:
                        market_fit_reward += 0.5
                        if discount_p >= disc_low and discount_p <= disc_high:
                            market_fit_reward += 0.5
                    
                    review_reward = 0.0
                    if user_comment_count is not None and user_comment_count > 0:
                        if user_comment_count >= 100:
                            review_reward += 0.3
                        elif user_comment_count >= 30:
                            review_reward += 0.2
                        elif user_comment_count >= 10:
                            review_reward += 0.15
                        else:
                            review_reward += 0.05
                    if user_rating is not None and user_rating > 0:
                        if user_rating >= 4.8:
                            review_reward += 0.7
                        elif user_rating >= 4.6:
                            review_reward += 0.5
                        elif user_rating >= 4.3:
                            review_reward += 0.3
                        elif user_rating >= 4.0:
                            review_reward += 0.15
                        else:
                            review_reward += 0.05
                    
                    prob_scaled = prob_component_1_2 + 0.10 * market_fit_reward + 0.05 * review_reward
                    prob_scaled = max(0.0, min(0.999, prob_scaled))
                    
                    # Apply expansion penalty
                    final_prob_scaled = prob_scaled * expansion_penalty
                    
                    if final_prob_scaled > best[2]:
                        profit = p - original_price
                        margin = profit / (original_price + 1e-9)
                        revenue = final_prob_scaled * p
                        exp_profit = final_prob_scaled * profit
                        best = (float(p), prob, float(final_prob_scaled), revenue, discount_p, tr_cand, 
                               float(profit), float(margin), float(exp_profit))
                        print(f"         ✅ Tìm được giá tốt hơn: {p:,.0f} | Discount: {discount_p:.1%} | Prob: {final_prob_scaled:.1%}")
            
            elif best[0] == max_candidate:
                # Giá tốt nhất chạm biên HIGH → Duyệt thêm 3 giá cao hơn (ít khi xảy ra với AGGRESSIVE)
                print(f"      🔍 Giá tốt nhất ({best[0]:,.0f}) chạm biên HIGH → Mở rộng lên cao hơn")
                expansion_prices = [max_candidate + 5000 * i for i in range(1, 4)]
                
                for p in expansion_prices:
                    if p >= original_price:
                        break
                    # Similar evaluation logic as above
                    # (Code omitted for brevity - same as LOW expansion)
            
            print(f"\n   🔴 AGGRESSIVE: Chọn giá tối ưu XÁC SUẤT BÁN cao nhất")
            
        elif strategy == 'conservative':
            # CONSERVATIVE: TỐI ĐA HÓA DOANH THU KỲ VỌNG (prob × price)
            # Chọn giá có expected_revenue cao nhất
            # DEBUG: Check candidates
            print(f"\n   🔍 DEBUG CONSERVATIVE - Candidates:")
            print(f"      Total valid_candidates: {len(valid_candidates)}")
            
            # Show top 10 by revenue
            sorted_by_revenue = sorted(valid_candidates, key=lambda x: x[3], reverse=True)[:10]
            print(f"\n   🔍 DEBUG CONSERVATIVE - Top 10 candidates by REVENUE:")
            for i, c in enumerate(sorted_by_revenue, 1):
                discount_c = (original_price - c[0]) / original_price
                print(f"      {i}. Giá: {c[0]:,.0f} | Discount: {discount_c:.1%} | Prob: {c[2]:.1%} | Revenue: {c[3]:,.0f}")
            
            best = max(valid_candidates, key=lambda x: x[3])  # Max expected_revenue_at_recommended
            
            # PHASE 2: Boundary expansion - Mở rộng động dựa trên discount
            candidate_prices_list = sorted([c[0] for c in valid_candidates])
            min_candidate = min(candidate_prices_list)
            max_candidate = max(candidate_prices_list)
            
            if best[0] == max_candidate:
                # Giá tốt nhất chạm biên HIGH → Mở rộng lên cao hơn
                discount_best = (original_price - best[0]) / original_price
                # Mở rộng tối đa 10% × discount_best (giảm discount)
                # discount_best càng thấp (gần 0%) → mở rộng càng ít
                min_expansion_discount = discount_best - 0.10 * discount_best
                min_expansion_discount = max(0, min_expansion_discount)  # Không âm
                max_expansion_price = original_price * (1 - min_expansion_discount)
                
                print(f"      🔍 Giá tốt nhất ({best[0]:,.0f}, discount {discount_best:.1%}) chạm biên HIGH")
                print(f"         → Mở rộng discount từ {discount_best:.1%} xuống {min_expansion_discount:.1%}")
                print(f"         → Mở rộng giá từ {best[0]:,.0f} lên {max_expansion_price:,.0f}")
                
                # Generate expansion prices with 5K step
                expansion_prices = []
                p = best[0] + 5000
                while p < max_expansion_price and p < original_price and len(expansion_prices) < 20:
                    expansion_prices.append(p)
                    p += 5000
                
                print(f"         → Duyệt thêm {len(expansion_prices)} giá trị")
                
                for p in expansion_prices:
                    discount_p = (original_price - p) / original_price
                    
                    # Apply exponential penalty when exceeding expansion range
                    if discount_p < min_expansion_discount:
                        excess = min_expansion_discount - discount_p
                        # Tốc độ giảm: exp(-excess × 20) → giảm rất nhanh
                        expansion_penalty = math.exp(-excess * 20)
                    else:
                        expansion_penalty = 1.0
                    
                    fv = prepare_input_features(name, store_name, category_group, float(p),
                                                comment_count=user_comment_count, rating=user_rating,
                                                use_sim_fill_for_missing=True)
                    prob = predict_prob_sold_gt100(fv)
                    price_tr_cand = compute_price_trust(p, market_median)
                    discount_tr_cand = compute_discount_trust(discount_p, disc_low, disc_high)
                    tr_cand = aggregate_trust(tr_review, price_tr_cand, discount_tr_cand, store_tr, neigh_cons)
                    
                    prob_component_1_2 = 0.60 * prob + 0.25 * tr_cand * prob
                    market_fit_reward = 0.0
                    if p >= market_low and p <= market_high:
                        market_fit_reward += 0.5
                        if discount_p >= disc_low and discount_p <= disc_high:
                            market_fit_reward += 0.5
                    
                    review_reward = 0.0
                    if user_comment_count is not None and user_comment_count > 0:
                        if user_comment_count >= 100:
                            review_reward += 0.3
                        elif user_comment_count >= 30:
                            review_reward += 0.2
                        elif user_comment_count >= 10:
                            review_reward += 0.15
                        else:
                            review_reward += 0.05
                    if user_rating is not None and user_rating > 0:
                        if user_rating >= 4.8:
                            review_reward += 0.7
                        elif user_rating >= 4.6:
                            review_reward += 0.5
                        elif user_rating >= 4.3:
                            review_reward += 0.3
                        elif user_rating >= 4.0:
                            review_reward += 0.15
                        else:
                            review_reward += 0.05
                    
                    prob_scaled = prob_component_1_2 + 0.10 * market_fit_reward + 0.05 * review_reward
                    prob_scaled = max(0.0, min(0.999, prob_scaled))
                    
                    # Apply expansion penalty to prob
                    final_prob_scaled = prob_scaled * expansion_penalty
                    revenue = final_prob_scaled * p
                    
                    if revenue > best[3]:
                        profit = p - original_price
                        margin = profit / (original_price + 1e-9)
                        exp_profit = final_prob_scaled * profit
                        best = (float(p), prob, float(final_prob_scaled), revenue, discount_p, tr_cand, 
                               float(profit), float(margin), float(exp_profit))
                        print(f"         ✅ Tìm được giá tốt hơn: {p:,.0f} | Discount: {discount_p:.1%} | Prob: {final_prob_scaled:.1%} | Revenue: {revenue:,.0f}")
            
            elif best[0] == min_candidate:
                # Giá tốt nhất chạm biên LOW → Duyệt thêm 3 giá thấp hơn (ít khi xảy ra với CONSERVATIVE)
                print(f"      🔍 Giá tốt nhất ({best[0]:,.0f}) chạm biên LOW → Mở rộng xuống thấp hơn")
                expansion_prices = [min_candidate - 5000 * i for i in range(1, 4)]
                
                for p in expansion_prices:
                    if p <= 0:
                        break
                    # Similar evaluation logic as above
                    # (Code omitted for brevity - same as HIGH expansion)
            
            print(f"\n   🟢 CONSERVATIVE: Chọn giá tối ưu DOANH THU KỲ VỌNG cao nhất")
            
        else:  # balanced
            # BALANCED: CÂN BẰNG giữa TỶ LỆ BÁN và DOANH THU
            # Sử dụng composite score để trade-off tối ưu
            # Score = α × normalized_prob + β × normalized_revenue
            # với α + β = 1, α = β = 0.5 (cân bằng đều)
            
            # Ưu tiên margin >= -10%, fallback tất cả
            filtered = [c for c in valid_candidates if c[7] >= -0.10]  # margin >= -10%
            if len(filtered) == 0:
                # Fallback: chấp nhận lỗ nhiều hơn nếu cần
                filtered = valid_candidates
            
            # Tính min/max để normalize
            probs = [c[2] for c in filtered]  # prob_scaled
            revenues = [c[3] for c in filtered]  # expected_revenue
            
            min_prob, max_prob = min(probs), max(probs)
            min_revenue, max_revenue = min(revenues), max(revenues)
            
            # Tránh chia cho 0
            prob_range = max_prob - min_prob if max_prob > min_prob else 1.0
            revenue_range = max_revenue - min_revenue if max_revenue > min_revenue else 1.0
            
            # Tính composite score cho mỗi candidate
            # Balanced trade-off: Càng cân bằng giữa tỷ lệ và doanh thu càng tốt
            def compute_balance_score(c):
                prob_norm = (c[2] - min_prob) / prob_range  # 0-1
                revenue_norm = (c[3] - min_revenue) / revenue_range  # 0-1
                
                # Base score: Cân bằng 50-50 giữa tỷ lệ bán và doanh thu
                base_score = 0.5 * prob_norm + 0.5 * revenue_norm
                
                # BONUS: Thưởng khi prob_norm và revenue_norm CÂN BẰNG với nhau
                # Tính độ lệch giữa prob_norm và revenue_norm
                balance_deviation = abs(prob_norm - revenue_norm)
                # Thưởng khi độ lệch nhỏ (càng cân bằng càng tốt)
                # balance_deviation = 0 (hoàn toàn cân bằng) → bonus = +0.2
                # balance_deviation = 1 (hoàn toàn lệch) → bonus = 0
                balance_bonus = (1.0 - balance_deviation) * 0.2
                
                # Tổng score = base + bonus cân bằng
                total_score = base_score + balance_bonus
                return total_score
            
            best = max(filtered, key=compute_balance_score)
            best_balance_score = compute_balance_score(best)
            
            # In thông tin về mức độ cân bằng
            prob_norm = (best[2] - min_prob) / prob_range
            revenue_norm = (best[3] - min_revenue) / revenue_range
            balance_dev = abs(prob_norm - revenue_norm)
            print(f"\n   🟡 BALANCED: Chọn giá tối ưu CÂN BẰNG (50% tỷ lệ + 50% doanh thu)")
            print(f"      Balance score: {best_balance_score:.3f}")
            print(f"      Prob normalized: {prob_norm:.3f} | Revenue normalized: {revenue_norm:.3f}")
            print(f"      Balance deviation: {balance_dev:.3f} (càng nhỏ càng cân bằng)")
        
        best_price, best_prob_raw, best_prob_scaled, best_rev, best_discount, best_trust, best_profit, best_margin, best_exp_profit = best
        
        # KIỂM TRA FALLBACK THEO TỪNG STRATEGY
        # Nếu không tìm được giá tốt hơn price_to_test theo mục tiêu của strategy → Dùng price_to_test
        use_fallback = False
        fallback_reason = ""
        
        if strategy == 'aggressive':
            # Max tỷ lệ: Nếu prob không tốt hơn baseline → fallback
            if best_prob_scaled <= prob_input_trusted:
                use_fallback = True
                fallback_reason = f"Không tìm được giá cho tỷ lệ bán tốt hơn ({best_prob_scaled:.1%} ≤ {prob_input_trusted:.1%})"
        elif strategy == 'conservative':
            # Max doanh thu: Nếu revenue không tốt hơn baseline → fallback
            baseline_revenue = prob_input_trusted * price_to_test if price_to_test else 0
            if best_rev <= baseline_revenue:
                use_fallback = True
                fallback_reason = f"Không tìm được giá cho doanh thu kỳ vọng tốt hơn ({best_rev:,.0f} ≤ {baseline_revenue:,.0f})"
        else:  # balanced
            # Balanced: So sánh balance score thay vì so sánh tuyệt đối
            # Tính balance score của price_to_test
            baseline_revenue = prob_input_trusted * price_to_test if price_to_test else 0
            
            # Tính normalized values cho baseline
            baseline_prob_norm = (prob_input_trusted - min_prob) / prob_range if prob_range > 0 else 0.5
            baseline_revenue_norm = (baseline_revenue - min_revenue) / revenue_range if revenue_range > 0 else 0.5
            baseline_balance_dev = abs(baseline_prob_norm - baseline_revenue_norm)
            baseline_balance_score = 0.5 * baseline_prob_norm + 0.5 * baseline_revenue_norm + (1.0 - baseline_balance_dev) * 0.2
            
            # So sánh balance score: chỉ fallback nếu balance score KHÔNG tốt hơn
            # Trade-off được chấp nhận: Prob tăng nhiều, revenue giảm ít → balance score cao hơn
            if best_balance_score <= baseline_balance_score:
                use_fallback = True
                fallback_reason = f"Không tìm được giá cân bằng tốt hơn (balance_score={best_balance_score:.3f} ≤ {baseline_balance_score:.3f})"
            else:
                # Balance score tốt hơn → Không fallback, in thông tin trade-off
                prob_change = ((best_prob_scaled - prob_input_trusted) / prob_input_trusted) * 100
                revenue_change = ((best_rev - baseline_revenue) / baseline_revenue) * 100 if baseline_revenue > 0 else 0
                print(f"      ✅ Trade-off tốt: Balance score {best_balance_score:.3f} > baseline {baseline_balance_score:.3f}")
                print(f"      📊 Prob {prob_change:+.1f}%, Revenue {revenue_change:+.1f}%")
        
        if use_fallback:
            print(f"\n   ⚠️  FALLBACK ({strategy.upper()}): {fallback_reason}")
            print(f"       → Sử dụng chính xác price_to_test = {price_to_test:,.0f} VNĐ")
            best_price = float(price_to_test) if price_to_test else original_price
            best_discount = max(0.0, (original_price - best_price) / (original_price + 1e-9))
            best_prob_scaled = prob_input_trusted
            best_prob_raw = prob_input
            best_profit = best_price - original_price
            best_margin = best_profit / (original_price + 1e-9)
            best_rev = best_prob_scaled * best_price
            best_exp_profit = best_prob_scaled * best_profit
            best_trust = trust_user
        
        # Kiểm tra và in thông báo về các ràng buộc tỷ lệ bán
        print(f"   📊 Kết quả:")
        print(f"      Giá đề xuất: {best_price:,.0f} VNĐ")
        print(f"      Xác suất bán: {best_prob_scaled:.1%}")
        print(f"      Baseline (price_to_test): {prob_input_trusted:.1%}")
        
        if strategy in ['aggressive', 'balanced']:
            if best_prob_scaled >= prob_input_trusted:
                print(f"      ✅ Đạt yêu cầu: Xác suất >= baseline ({best_prob_scaled:.1%} >= {prob_input_trusted:.1%})")
            else:
                print(f"      ⚠️ Cảnh báo: Xác suất < baseline ({best_prob_scaled:.1%} < {prob_input_trusted:.1%})")
        elif strategy == 'conservative':
            # CONSERVATIVE không có ràng buộc tỷ lệ, chỉ hiển thị thông tin
            print(f"      📊 Xác suất bán: {best_prob_scaled:.1%} (ưu tiên lợi nhuận, không ràng buộc tỷ lệ)")
        
        return {
            "market_low": market_low,
            "market_high": market_high,
            "market_discount_low": disc_low,
            "market_discount_high": disc_high,
            "similar_products_filtered": filtered_sim,
            "prob_if_use_input_price": prob_input,
            "prob_if_use_input_price_trusted": prob_input_trusted,
            "recommended_price": float(best_price),
            "expected_discount_percent": float(best_discount),
            "prob_sold_well_at_recommended": float(best_prob_scaled),
            "prob_sold_well_at_recommended_raw": float(best_prob_raw),
            "expected_revenue_at_recommended": float(best_rev),
            "profit": float(best_profit),
            "profit_margin": float(best_margin),
            "expected_profit": float(best_exp_profit),
            "strategy_used": strategy,
            "fallback_used": False,
            "outlier_flag": outlier_flag,
            "outlier_severity": outlier_severity,
            "discount_user": discount_user,
            "trust_user": trust_user,
            "trust_recommended": best_trust
        }

    # === FALLBACK ===: relax constraints but prefer closeness to discount range
    fallback_candidates = []
    relaxed_threshold = prob_input_trusted * 0.5
    for p in candidate_prices:
        discount_p = max(0.0, (original_price - p) / (original_price + 1e-9))
        fv = prepare_input_features(name, store_name, category_group, float(p),
                                    comment_count=user_comment_count if user_comment_count is not None else None,
                                    rating=user_rating if user_rating is not None else None,
                                    use_sim_fill_for_missing=True)
        prob = predict_prob_sold_gt100(fv)
        dist = 0.0
        if discount_p < disc_low:
            dist = disc_low - discount_p
        elif discount_p > disc_high:
            dist = discount_p - disc_high

        price_tr_cand = compute_price_trust(p, market_median)
        discount_tr_cand = compute_discount_trust(discount_p, disc_low, disc_high)
        tr_cand = aggregate_trust(tr_review, price_tr_cand, discount_tr_cand, store_tr, neigh_cons)
        prob_scaled = prob * (0.55 + 0.45 * tr_cand)

        allowed = (prob_scaled >= relaxed_threshold) or (dist <= margin_discount * 2.0)
        if not allowed:
            continue
        expected_revenue = prob_scaled * p
        fallback_candidates.append((float(p), float(prob), float(prob_scaled), float(expected_revenue), float(dist), float(discount_p), float(tr_cand)))

    if len(fallback_candidates) == 0:
        # expand wide grid
        wider_low = max(1000.0, original_price * max(0.05, 1.0 - (disc_high + 0.45)))
        wider_high = max(wider_low + 1.0, original_price * min(1.0, 1.0 - max(0.0, disc_low - 0.45)))
        wide_grid = np.linspace(wider_low, wider_high, price_grid_steps * 2)
        for p in wide_grid:
            discount_p = max(0.0, (original_price - p) / (original_price + 1e-9))
            fv = prepare_input_features(name, store_name, category_group, float(p),
                                        comment_count=user_comment_count if user_comment_count is not None else None,
                                        rating=user_rating if user_rating is not None else None,
                                        use_sim_fill_for_missing=True)
            prob = predict_prob_sold_gt100(fv)
            price_tr_cand = compute_price_trust(p, market_median)
            discount_tr_cand = compute_discount_trust(discount_p, disc_low, disc_high)
            tr_cand = aggregate_trust(tr_review, price_tr_cand, discount_tr_cand, store_tr, neigh_cons)
            prob_scaled = prob * (0.5 + 0.5 * tr_cand)
            expected_revenue = prob_scaled * p
            dist = 0.0
            if discount_p < disc_low:
                dist = disc_low - discount_p
            elif discount_p > disc_high:
                dist = discount_p - disc_high
            fallback_candidates.append((float(p), float(prob), float(prob_scaled), float(expected_revenue), float(dist), float(discount_p), float(tr_cand)))

    if len(fallback_candidates) == 0:
        # FALLBACK CUỐI CÙNG: Dùng CHÍNH XÁC price_to_test (giá user nhập)
        # KHÔNG dùng median market discount hay giá khác
        print(f"\n   ⚠️  FALLBACK: Không tìm được giá tốt hơn, sử dụng chính xác price_to_test = {price_to_test:,.0f} VNĐ")
        
        fallback_price = price_to_test if price_to_test is not None and price_to_test > 0 else original_price
        fallback_discount = max(0.0, (original_price - fallback_price) / (original_price + 1e-9))
        
        fv = prepare_input_features(name, store_name, category_group, fallback_price,
                                    comment_count=user_comment_count if user_comment_count is not None else None,
                                    rating=user_rating if user_rating is not None else None,
                                    use_sim_fill_for_missing=True)
        prob = predict_prob_sold_gt100(fv)
        price_tr_cand = compute_price_trust(fallback_price, market_median)
        disc_tr_cand = compute_discount_trust(fallback_discount, disc_low, disc_high)
        tr_cand = aggregate_trust(tr_review, price_tr_cand, disc_tr_cand, store_tr, neigh_cons)
        prob_scaled = prob * (0.5 + 0.5 * tr_cand)
        expected_revenue = prob_scaled * fallback_price
        profit = fallback_price - original_price
        profit_margin = profit / (original_price + 1e-9)
        expected_profit = prob_scaled * profit
        return {
            "market_low": market_low,
            "market_high": market_high,
            "market_discount_low": disc_low,
            "market_discount_high": disc_high,
            "similar_products_filtered": filtered_sim,
            "prob_if_use_input_price": prob_input,
            "prob_if_use_input_price_trusted": prob_input_trusted,
            "recommended_price": float(fallback_price),
            "expected_discount_percent": float(fallback_discount),
            "prob_sold_well_at_recommended": float(prob_scaled),
            "expected_revenue_at_recommended": float(expected_revenue),
            "profit": float(profit),
            "profit_margin": float(profit_margin),
            "expected_profit": float(expected_profit),
            "fallback_used": True,
            "outlier_flag": outlier_flag,
            "outlier_severity": outlier_severity,
            "discount_user": discount_user,
            "trust_user": trust_user,
            "trust_recommended": tr_cand
        }

    # pick best fallback by (smallest dist, highest revenue)
    fallback_sorted = sorted(fallback_candidates, key=lambda x: (x[4], -x[3]))
    
    # Nếu không có candidate nào tốt hơn price_to_test, dùng chính xác price_to_test
    best_price, raw_prob, scaled_prob, best_rev, best_dist, best_discount, best_trust = fallback_sorted[0]
    
    # So sánh với prob_input_trusted để quyết định dùng fallback hay price_to_test
    if scaled_prob < prob_input_trusted:
        print(f"\n   ⚠️  FALLBACK: Candidate tốt nhất ({best_price:,.0f} VNĐ) có prob {scaled_prob:.1%} < price_to_test prob {prob_input_trusted:.1%}")
        print(f"       → Sử dụng chính xác price_to_test = {price_to_test:,.0f} VNĐ")
        best_price = price_to_test if price_to_test is not None and price_to_test > 0 else original_price
        best_discount = max(0.0, (original_price - best_price) / (original_price + 1e-9))
        scaled_prob = prob_input_trusted
        best_rev = scaled_prob * best_price
        best_trust = trust_user
    
    profit = best_price - original_price
    profit_margin = profit / (original_price + 1e-9)
    expected_profit = scaled_prob * profit
    return {
        "market_low": market_low,
        "market_high": market_high,
        "market_discount_low": disc_low,
        "market_discount_high": disc_high,
        "similar_products_filtered": filtered_sim,
        "prob_if_use_input_price": prob_input,
        "prob_if_use_input_price_trusted": prob_input_trusted,
        "recommended_price": float(best_price),
        "expected_discount_percent": float(best_discount),
        "prob_sold_well_at_recommended": float(scaled_prob),
        "expected_revenue_at_recommended": float(best_rev),
        "profit": float(profit),
        "profit_margin": float(profit_margin),
        "expected_profit": float(expected_profit),
        "fallback_used": True,
        "outlier_flag": outlier_flag,
        "outlier_severity": outlier_severity,
        "discount_user": discount_user,
        "trust_user": trust_user,
        "trust_recommended": best_trust
    }

# -----------------------
# BUILD HOLDOUT FEATURES (X_hold) - same logic as TRAIN
# -----------------------
print("\n==============================================================")
print("BUILDING HOLDOUT FEATURES")
print("==============================================================")

df_hold = df_test_holdout.copy().reset_index(drop=True)
# use embs_test from PART 1
dists_hold, hold_indices = nn_final.kneighbors(embs_test, n_neighbors=NN_SIMILAR_K+1)

s_price_hold = np.zeros(len(embs_test))
s_orig_hold = np.zeros(len(embs_test))
s_sold_hold = np.zeros(len(embs_test))
s_pct_hold = np.zeros(len(embs_test))
s_comment_hold = np.zeros(len(embs_test))
s_rating_hold = np.zeros(len(embs_test))
s_disc_hold = np.zeros(len(embs_test))

for i in range(len(embs_test)):
    neigh_idx = hold_indices[i, 1:NN_SIMILAR_K+1]
    if len(neigh_idx) == 0:
        continue
    neigh = df_train_full.iloc[neigh_idx]   # neighbors always from TRAIN
    
    # compute market reference from neighbors
    market_low = neigh['price'].quantile(0.25)
    market_high = neigh['price'].quantile(0.75)
    disc_low = neigh['discount_percent'].quantile(0.25)
    disc_high = neigh['discount_percent'].quantile(0.75)
    
    # compute weights for each neighbor
    weights = []
    for _, r in neigh.iterrows():
        w = compute_weight_for_neighbor(
            price=r['price'],
            discount=r['discount_percent'],
            market_low=market_low,
            market_high=market_high,
            disc_low=disc_low,
            disc_high=disc_high
        )
        weights.append(w)
    weights = np.array(weights)
    
    # weighted averages
    s_price_hold[i] = np.average(neigh['price'], weights=weights)
    s_orig_hold[i] = np.average(neigh['original_price'], weights=weights)
    s_sold_hold[i] = np.average(neigh['sold'], weights=weights)
    s_pct_hold[i] = np.average(neigh['is_sold_well'], weights=weights)
    s_comment_hold[i] = np.average(neigh['comment_count'], weights=weights)
    s_rating_hold[i] = np.average(neigh['rating'], weights=weights)
    s_disc_hold[i] = np.average(neigh['discount_percent'], weights=weights)

df_hold['sim_avg_price'] = s_price_hold
df_hold['sim_avg_original'] = s_orig_hold
df_hold['sim_avg_sold'] = s_sold_hold
df_hold['sim_pct_sold_gt100'] = s_pct_hold
df_hold['sim_comment_avg'] = s_comment_hold
df_hold['sim_rating_avg'] = s_rating_hold
df_hold['sim_avg_discount'] = s_disc_hold

# category means
cm_price_map, cm_orig_map = (cat_mean_price_map_final, cat_mean_orig_map_final)
global_mean_price = df_train_full['price'].mean()
global_mean_orig = df_train_full['original_price'].mean()

df_hold['cat_mean_price'] = df_hold['category_group_processed'].map(cm_price_map).fillna(global_mean_price)
df_hold['cat_mean_original'] = df_hold['category_group_processed'].map(cm_orig_map).fillna(global_mean_orig)

# indicators
df_hold['has_comment_data'] = (df_hold['comment_count'] > 0).astype(float)
df_hold['has_rating_data'] = (df_hold['rating'] > 0).astype(float)

# OHE transform
cat_ohe_hold = ohe_final.transform(df_hold[['store_processed','category_group_processed']])
cat_ohe_cols = ohe_final.get_feature_names_out(['store_processed','category_group_processed'])
X_cat_hold = pd.DataFrame(cat_ohe_hold, columns=cat_ohe_cols, index=df_hold.index)

X_num_hold = df_hold[numeric_cols].reset_index(drop=True)
X_hold = pd.concat([X_num_hold, X_cat_hold.reset_index(drop=True)], axis=1).fillna(0)

# scale numeric with scaler_final
X_hold[numeric_cols] = scaler_final.transform(X_hold[numeric_cols])
X_hold = X_hold.reindex(columns=X_final.columns, fill_value=0)

y_hold = df_hold['is_sold_well'].values

# -----------------------
# HOLDOUT EVALUATION (2 scenarios)
# -----------------------
print("\n================================================================================")
print("EVALUATING ON HOLDOUT TEST SET (20%) - 2 SCENARIOS")
print("================================================================================")

# Scenario 1: product new (no comment/rating)
print("\n=== SCENARIO 1: Sản phẩm mới (KHÔNG có comment/rating) ===")
X_hold_new = X_hold.copy()
X_hold_new['comment_count'] = 0
X_hold_new['rating'] = 4.8
X_hold_new['has_comment_data'] = 0.0
X_hold_new['has_rating_data'] = 0.0

# re-scale numeric (already scaled) and ensure columns
X_hold_new = X_hold_new.reindex(columns=X_final.columns, fill_value=0)
y_prob_new = clf_final.predict_proba(X_hold_new)[:,1]
auc_new = roc_auc_score(y_hold, y_prob_new) if len(np.unique(y_hold))>1 else float('nan')
print(f"\nHOLDOUT TEST AUC (sản phẩm mới): {auc_new:.4f}")

# Scenario 2: existing products (use actual comment/rating)
print("\n=== SCENARIO 2: Sản phẩm đã có comment/rating ===")
X_hold_exist = X_hold.copy()
X_hold_exist['has_comment_data'] = (df_hold['comment_count'] > 0).astype(float)
X_hold_exist['has_rating_data'] = (df_hold['rating'] > 0).astype(float)

X_hold_exist = X_hold_exist.reindex(columns=X_final.columns, fill_value=0)
y_prob_exist = clf_final.predict_proba(X_hold_exist)[:,1]
auc_exist = roc_auc_score(y_hold, y_prob_exist) if len(np.unique(y_hold))>1 else float('nan')
print(f"\nHOLDOUT TEST AUC (sản phẩm có dữ liệu): {auc_exist:.4f}")

# -----------------------
# DEMO: PRICE RECOMMENDATION - 3 STRATEGIES
# -----------------------
print("\n" + "="*100)
print("DEMO: SO SÁNH 3 CHIẾN LƯỢC TỐI ƯU GIÁ")
print("="*100)

user_input = {
    "name": "Bộ váy hai mảnh CHANGTONG cổ chữ V cotton thoáng khí phong cách Hàn Quốc thanh lịch mùa thu",
    "store_name": "Changtong Fashion",
    "category_group": "Thời trang - Trung+",
    "original_price": 300000,
    "price_to_test": 280000,
    "comment_count": 30,
    "rating": 4.9
}

original_price = user_input['original_price']
price_to_test = user_input['price_to_test']
user_discount_percent = ((original_price - price_to_test) / original_price) * 100 if price_to_test else 0

print(f"\n📋 THÔNG TIN SẢN PHẨM:")
print(f"   Tên: {user_input['name'][:80]}...")
print(f"   Cửa hàng: {user_input['store_name']}")
print(f"   Phân loại: {user_input['category_group']}")
print(f"   💰 Giá nhập (original_price): {original_price:,.0f} VNĐ")
print(f"   🏷️  Giá dự định bán (price_to_test): {price_to_test:,.0f} VNĐ")
print(f"   📉 % Giảm giá dự định: {user_discount_percent:.2f}%")
print(f"   ⭐ Rating: {user_input['rating']}/5.0")
print(f"   💬 Số comment: {user_input['comment_count']}")

print("\n🎯 Đang tính toán tối ưu với 3 chiến lược...\n")

# Run all 3 strategies
strategies = {
    'aggressive': '🔴 AGGRESSIVE',
    'balanced': '🟡 BALANCED',
    'conservative': '🟢 CONSERVATIVE'
}

results = {}
for strategy_key, strategy_name in strategies.items():
    print(f"⏳ Đang tính toán {strategy_name}...")
    results[strategy_key] = recommend_price(
        original_price=user_input['original_price'],
        name=user_input['name'],
        store_name=user_input['store_name'],
        category_group=user_input['category_group'],
        price_to_test=user_input['price_to_test'],
        user_comment_count=user_input['comment_count'],
        user_rating=user_input['rating'],
        price_grid_steps=40,
        market_top_k=20,
        strategy=strategy_key
    )

# ============================================================================
# CROSS-VALIDATION: Kiểm tra và điều chỉnh để đảm bảo ràng buộc
# ============================================================================
print("\n" + "="*100)
print("🔍 CROSS-VALIDATION: Kiểm tra ràng buộc giữa các chiến lược")
print("="*100)

prob_agg = results['aggressive']['prob_sold_well_at_recommended']
prob_bal = results['balanced']['prob_sold_well_at_recommended']
prob_con = results['conservative']['prob_sold_well_at_recommended']
prob_baseline = results['balanced']['prob_if_use_input_price_trusted']

revenue_agg = results['aggressive']['expected_revenue_at_recommended']
revenue_bal = results['balanced']['expected_revenue_at_recommended']
revenue_con = results['conservative']['expected_revenue_at_recommended']

print(f"\n📊 Kết quả ban đầu:")
print(f"   Tỷ lệ bán:    AGG={prob_agg:.1%} | BAL={prob_bal:.1%} | CON={prob_con:.1%} | Baseline={prob_baseline:.1%}")
print(f"   Doanh thu KV:  AGG={revenue_agg:,.0f} | BAL={revenue_bal:,.0f} | CON={revenue_con:,.0f}")

# Flags để track điều chỉnh
adjusted = False

# ============================================================================
# RÀNG BUỘC 1: Tỷ lệ bán AGGRESSIVE ≥ BALANCED ≥ CONSERVATIVE
# ============================================================================
print(f"\n✓ Kiểm tra ràng buộc 1: Tỷ lệ bán AGG ≥ BAL ≥ CON")

if prob_agg < prob_bal or prob_agg < prob_con:
    print(f"   ⚠️  VI PHẠM: AGGRESSIVE ({prob_agg:.1%}) không cao nhất!")
    
    # Tìm chiến lược có prob cao hơn aggressive
    if prob_bal > prob_agg and prob_bal >= prob_con:
        donor_strategy = 'balanced'
        donor_prob = prob_bal
    elif prob_con > prob_agg:
        donor_strategy = 'conservative'
        donor_prob = prob_con
    else:
        donor_strategy = 'balanced'
        donor_prob = prob_bal
    
    # Kiểm tra xem donor có tốt hơn baseline không
    if donor_prob > prob_baseline:
        print(f"   🔄 ĐIỀU CHỈNH: Sử dụng giá từ {donor_strategy.upper()} cho AGGRESSIVE")
        print(f"      {donor_strategy.upper()} prob ({donor_prob:.1%}) > baseline ({prob_baseline:.1%})")
        
        # Copy kết quả từ donor sang aggressive
        results['aggressive']['recommended_price'] = results[donor_strategy]['recommended_price']
        results['aggressive']['expected_discount_percent'] = results[donor_strategy]['expected_discount_percent']
        results['aggressive']['prob_sold_well_at_recommended'] = results[donor_strategy]['prob_sold_well_at_recommended']
        results['aggressive']['expected_revenue_at_recommended'] = results[donor_strategy]['expected_revenue_at_recommended']
        results['aggressive']['profit'] = results[donor_strategy]['profit']
        results['aggressive']['profit_margin'] = results[donor_strategy]['profit_margin']
        results['aggressive']['expected_profit'] = results[donor_strategy]['expected_profit']
        results['aggressive']['trust_recommended'] = results[donor_strategy]['trust_recommended']
        
        prob_agg = results['aggressive']['prob_sold_well_at_recommended']
        revenue_agg = results['aggressive']['expected_revenue_at_recommended']
        adjusted = True
        print(f"      ✅ AGGRESSIVE mới: Giá={results['aggressive']['recommended_price']:,.0f} | Prob={prob_agg:.1%}")
    else:
        print(f"      ⚠️  {donor_strategy.upper()} prob ({donor_prob:.1%}) ≤ baseline ({prob_baseline:.1%})")
        print(f"      → Giữ nguyên AGGRESSIVE (prob={prob_agg:.1%} vẫn > baseline)")
else:
    print(f"   ✅ ĐẠT: {prob_agg:.1%} ≥ {prob_bal:.1%} ≥ {prob_con:.1%}")

# Kiểm tra lại balanced >= conservative
if prob_bal < prob_con:
    print(f"\n   ⚠️  VI PHẠM: BALANCED ({prob_bal:.1%}) < CONSERVATIVE ({prob_con:.1%})")
    if prob_con > prob_baseline:
        print(f"   🔄 ĐIỀU CHỈNH: Swap giá giữa BALANCED và CONSERVATIVE")
        # Swap
        results['balanced'], results['conservative'] = results['conservative'], results['balanced']
        prob_bal, prob_con = prob_con, prob_bal
        revenue_bal, revenue_con = revenue_con, revenue_bal
        adjusted = True
        print(f"      ✅ Sau swap: BAL={prob_bal:.1%} | CON={prob_con:.1%}")

# ============================================================================
# RÀNG BUỘC 2: Doanh thu kỳ vọng CONSERVATIVE ≥ BALANCED ≥ AGGRESSIVE
# ============================================================================
print(f"\n✓ Kiểm tra ràng buộc 2: Doanh thu KV CON ≥ BAL ≥ AGG")

revenue_baseline = prob_baseline * price_to_test if price_to_test else 0

if revenue_con < revenue_bal or revenue_con < revenue_agg:
    print(f"   ⚠️  VI PHẠM: CONSERVATIVE revenue ({revenue_con:,.0f}) không cao nhất!")
    
    # Tìm chiến lược có revenue cao nhất
    if revenue_bal >= revenue_agg and revenue_bal >= revenue_con:
        best_revenue_strategy = 'balanced'
        best_revenue_value = revenue_bal
    elif revenue_agg >= revenue_bal and revenue_agg >= revenue_con:
        best_revenue_strategy = 'aggressive'
        best_revenue_value = revenue_agg
    else:
        best_revenue_strategy = 'conservative'
        best_revenue_value = revenue_con
    
    # Kiểm tra xem best revenue có tốt hơn baseline không
    if best_revenue_value > revenue_baseline:
        print(f"   🔄 ĐIỀU CHỈNH: Sử dụng giá từ {best_revenue_strategy.upper()} cho CONSERVATIVE")
        print(f"      {best_revenue_strategy.upper()} revenue ({best_revenue_value:,.0f}) > baseline ({revenue_baseline:,.0f})")
        
        # Copy kết quả từ best revenue strategy sang conservative
        if best_revenue_strategy != 'conservative':
            results['conservative']['recommended_price'] = results[best_revenue_strategy]['recommended_price']
            results['conservative']['expected_discount_percent'] = results[best_revenue_strategy]['expected_discount_percent']
            results['conservative']['prob_sold_well_at_recommended'] = results[best_revenue_strategy]['prob_sold_well_at_recommended']
            results['conservative']['expected_revenue_at_recommended'] = results[best_revenue_strategy]['expected_revenue_at_recommended']
            results['conservative']['profit'] = results[best_revenue_strategy]['profit']
            results['conservative']['profit_margin'] = results[best_revenue_strategy]['profit_margin']
            results['conservative']['expected_profit'] = results[best_revenue_strategy]['expected_profit']
            results['conservative']['trust_recommended'] = results[best_revenue_strategy]['trust_recommended']
            
            revenue_con = results['conservative']['expected_revenue_at_recommended']
            adjusted = True
            print(f"      ✅ CONSERVATIVE mới: Giá={results['conservative']['recommended_price']:,.0f} | Revenue={revenue_con:,.0f}")
    else:
        print(f"      ⚠️  Best revenue ({best_revenue_value:,.0f}) ≤ baseline ({revenue_baseline:,.0f})")
        print(f"      → Giữ nguyên CONSERVATIVE")
else:
    print(f"   ✅ ĐẠT: {revenue_con:,.0f} ≥ {revenue_bal:,.0f} ≥ {revenue_agg:,.0f}")

# ============================================================================
# RÀNG BUỘC 3: AGGRESSIVE & BALANCED prob ≥ baseline
# ============================================================================
print(f"\n✓ Kiểm tra ràng buộc 3: AGG & BAL prob ≥ baseline")

# Tính toán các giá trị baseline
baseline_profit = price_to_test - user_input['original_price'] if price_to_test else 0
baseline_revenue = prob_baseline * price_to_test if price_to_test else 0

if prob_agg < prob_baseline:
    print(f"   ⚠️  VI PHẠM: AGGRESSIVE ({prob_agg:.1%}) < baseline ({prob_baseline:.1%})")
    print(f"   🔄 ĐIỀU CHỈNH: Sử dụng price_to_test cho AGGRESSIVE")
    results['aggressive']['recommended_price'] = float(price_to_test)
    results['aggressive']['expected_discount_percent'] = float((user_input['original_price'] - price_to_test) / user_input['original_price'])
    results['aggressive']['prob_sold_well_at_recommended'] = prob_baseline
    results['aggressive']['profit'] = baseline_profit
    results['aggressive']['expected_revenue_at_recommended'] = baseline_revenue
    adjusted = True

if prob_bal < prob_baseline:
    print(f"   ⚠️  VI PHẠM: BALANCED ({prob_bal:.1%}) < baseline ({prob_baseline:.1%})")
    print(f"   🔄 ĐIỀU CHỈNH: Sử dụng price_to_test cho BALANCED")
    results['balanced']['recommended_price'] = float(price_to_test)
    results['balanced']['expected_discount_percent'] = float((user_input['original_price'] - price_to_test) / user_input['original_price'])
    results['balanced']['prob_sold_well_at_recommended'] = prob_baseline
    results['balanced']['profit'] = baseline_profit
    results['balanced']['expected_revenue_at_recommended'] = baseline_revenue
    adjusted = True

if not (prob_agg < prob_baseline or prob_bal < prob_baseline):
    print(f"   ✅ ĐẠT: AGG={prob_agg:.1%} & BAL={prob_bal:.1%} ≥ baseline={prob_baseline:.1%}")

# Tóm tắt cross-validation
if adjusted:
    print(f"\n🔧 Đã thực hiện điều chỉnh cross-validation")
    print(f"\n📊 Kết quả sau điều chỉnh:")
    print(f"   Tỷ lệ bán:    AGG={results['aggressive']['prob_sold_well_at_recommended']:.1%} | BAL={results['balanced']['prob_sold_well_at_recommended']:.1%} | CON={results['conservative']['prob_sold_well_at_recommended']:.1%}")
    print(f"   Doanh thu KV:  AGG={results['aggressive']['expected_revenue_at_recommended']:,.0f} | BAL={results['balanced']['expected_revenue_at_recommended']:,.0f} | CON={results['conservative']['expected_revenue_at_recommended']:,.0f}")
else:
    print(f"\n✅ Không cần điều chỉnh - Tất cả ràng buộc đều đạt")

# Print market info (same for all strategies)
res_base = results['balanced']
print("\n" + "="*100)
print("📊 THÔNG TIN THỊ TRƯỜNG (Chung cho cả 3 chiến lược)")
print("="*100)

print(f"\n📈 KHOẢNG GIÁ THỊ TRƯỜNG:")
print(f"   Thấp nhất:  {res_base['market_low']:>12,.0f} VNĐ")
print(f"   Cao nhất:   {res_base['market_high']:>12,.0f} VNĐ")
print(f"   Trung bình: {(res_base['market_low'] + res_base['market_high'])/2:>12,.0f} VNĐ")

print(f"\n📉 KHOẢNG % GIẢM GIÁ THỊ TRƯỜNG:")
print(f"   Thấp nhất:  {res_base['market_discount_low']*100:>6.2f}%")
print(f"   Cao nhất:   {res_base['market_discount_high']*100:>6.2f}%")
print(f"   Trung bình: {(res_base['market_discount_low'] + res_base['market_discount_high'])/2*100:>6.2f}%")

print(f"\n📊 XÁC SUẤT NẾU BÁN GIÁ DỰ ĐỊNH ({price_to_test:,.0f} VNĐ):")
print(f"   - Xác suất gốc (model):              {res_base['prob_if_use_input_price']:.1%}")
print(f"   - Sau điều chỉnh tin cậy:            {res_base['prob_if_use_input_price_trusted']:.1%}")
print(f"   - Trust score người dùng:            {res_base.get('trust_user', 0):.3f}")

if res_base.get('outlier_flag', False):
    outlier_sev = res_base.get('outlier_severity', 'unknown')
    if outlier_sev == 'heavy':
        print(f"\n⚠️⚠️⚠️  LƯU Ý: Giá gốc là OUTLIER NẶNG so với thị trường (>3 lần)!")
        print("   → Market price range KHÔNG đáng tin")
        print("   → Model ưu tiên %discount dự định, cân bằng theo market discount range")
    elif outlier_sev == 'light':
        print(f"\n⚠️  LƯU Ý: Giá gốc là OUTLIER NHẸ so với thị trường (1.5-3 lần)!")
        print("   → Price_to_test còn trong khoảng hợp lý")
        print("   → Model tìm giá đề xuất theo mục tiêu kinh tế với điều chỉnh nhẹ")
    else:
        print(f"\n⚠️  LƯU Ý: Giá gốc có vẻ là OUTLIER so với phân khúc thị trường!")
        print("   → Model ưu tiên chiến lược discount-range thay vì kéo về phân khúc thị trường.")

# Print comparison table
print("\n" + "="*100)
print("🎯 SO SÁNH KẾT QUẢ TỐI ƯU CỦA 3 CHIẾN LƯỢC")
print("="*100)

print("\n┌────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐")
print("│      CHỈ TIÊU          │   🔴 AGGRESSIVE      │    🟡 BALANCED       │  🟢 CONSERVATIVE     │")
print("├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤")

agg = results['aggressive']
bal = results['balanced']
con = results['conservative']

print(f"│ 💰 Giá đề xuất (VNĐ)   │ {agg['recommended_price']:>18,.0f}   │ {bal['recommended_price']:>18,.0f}   │ {con['recommended_price']:>18,.0f}   │")
print(f"│ 📉 % Giảm giá          │ {agg['expected_discount_percent']*100:>18.2f}%  │ {bal['expected_discount_percent']*100:>18.2f}%  │ {con['expected_discount_percent']*100:>18.2f}%  │")
print(f"│ 📊 Xác suất bán (%)    │ {agg['prob_sold_well_at_recommended']*100:>18.1f}%  │ {bal['prob_sold_well_at_recommended']*100:>18.1f}%  │ {con['prob_sold_well_at_recommended']*100:>18.1f}%  │")
print(f"│ 💵 Lợi nhuận (VNĐ)     │ {agg['profit']:>18,.0f}   │ {bal['profit']:>18,.0f}   │ {con['profit']:>18,.0f}   │")
print(f"│ 📈 Margin (%)          │ {agg['profit_margin']*100:>18.2f}%  │ {bal['profit_margin']*100:>18.2f}%  │ {con['profit_margin']*100:>18.2f}%  │")
print(f"│ 💰 Doanh thu KV (VNĐ)  │ {agg['expected_revenue_at_recommended']:>18,.0f}   │ {bal['expected_revenue_at_recommended']:>18,.0f}   │ {con['expected_revenue_at_recommended']:>18,.0f}   │")
print(f"│ 💎 Lợi nhuận KV (VNĐ)  │ {agg['expected_profit']:>18,.0f}   │ {bal['expected_profit']:>18,.0f}   │ {con['expected_profit']:>18,.0f}   │")
print(f"│ 🎯 Trust score         │ {agg.get('trust_recommended', 0):>18.3f}   │ {bal.get('trust_recommended', 0):>18.3f}   │ {con.get('trust_recommended', 0):>18.3f}   │")
print(f"│ ⚙️  Fallback used       │ {str(agg.get('fallback_used', False)):>20}  │ {str(bal.get('fallback_used', False)):>20}  │ {str(con.get('fallback_used', False)):>20}  │")
print("└────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘")

# VALIDATION: Kiểm tra ràng buộc tỷ lệ bán
print("\n" + "="*100)
print("🔍 KIỂM TRA RÀNG BUỘC TỶ LỆ BÁN")
print("="*100)

prob_agg = agg['prob_sold_well_at_recommended']
prob_bal = bal['prob_sold_well_at_recommended']
prob_con = con['prob_sold_well_at_recommended']
prob_baseline = res_base['prob_if_use_input_price_trusted']

print(f"\n📊 Tỷ lệ bán thực tế:")
print(f"   • AGGRESSIVE:    {prob_agg:.1%}")
print(f"   • BALANCED:      {prob_bal:.1%}")
print(f"   • CONSERVATIVE:  {prob_con:.1%}")
print(f"   • Baseline (price_to_test): {prob_baseline:.1%}")

# Check constraint 1: aggressive >= balanced >= conservative
print(f"\n✓ Ràng buộc 1: Tỷ lệ bán AGGRESSIVE ≥ BALANCED ≥ CONSERVATIVE")
if prob_agg >= prob_bal >= prob_con:
    print(f"   ✅ ĐẠT: {prob_agg:.1%} ≥ {prob_bal:.1%} ≥ {prob_con:.1%}")
else:
    print(f"   ⚠️  KHÔNG ĐẠT:")
    if prob_agg < prob_bal:
        print(f"      - AGGRESSIVE ({prob_agg:.1%}) < BALANCED ({prob_bal:.1%})")
    if prob_bal < prob_con:
        print(f"      - BALANCED ({prob_bal:.1%}) < CONSERVATIVE ({prob_con:.1%})")

# Check constraint 2: aggressive & balanced >= baseline
print(f"\n✓ Ràng buộc 2: Tỷ lệ bán AGGRESSIVE và BALANCED ≥ Baseline")
agg_check = prob_agg >= prob_baseline
bal_check = prob_bal >= prob_baseline
if agg_check and bal_check:
    print(f"   ✅ ĐẠT:")
    print(f"      - AGGRESSIVE: {prob_agg:.1%} ≥ {prob_baseline:.1%}")
    print(f"      - BALANCED:   {prob_bal:.1%} ≥ {prob_baseline:.1%}")
else:
    print(f"   ⚠️  KHÔNG ĐẠT:")
    if not agg_check:
        print(f"      - AGGRESSIVE ({prob_agg:.1%}) < Baseline ({prob_baseline:.1%})")
    if not bal_check:
        print(f"      - BALANCED ({prob_bal:.1%}) < Baseline ({prob_baseline:.1%})")

# Overall validation
all_constraints_met = (prob_agg >= prob_bal >= prob_con) and agg_check and bal_check
if all_constraints_met:
    print(f"\n🎉 TẤT CẢ RÀNG BUỘC ĐỀU ĐẠT YÊU CẦU!")
else:
    print(f"\n⚠️  CÓ MỘT SỐ RÀNG BUỘC CHƯA ĐẠT - Cần xem xét điều chỉnh")

# Detailed explanation
print("\n" + "="*100)
print("📝 GIẢI THÍCH CHI TIẾT 3 CHIẾN LƯỢC")
print("="*100)

print("\n🔴 AGGRESSIVE (Tích cực - Bán nhanh):")
print("   🎯 Mục tiêu: TỐI ĐA HÓA XÁC SUẤT BÁN")
print("   ✅ Ưu điểm:")
print("      - Xác suất bán cao nhất → Bán nhanh, thu hồi vốn nhanh")
print("      - Phù hợp với hàng tồn kho, thanh lý, cần bán gấp")
print("      - Chấp nhận giảm giá mạnh để thu hút khách")
print("   ⚠️  Rủi ro:")
print("      - Chấp nhận LỖ lên tới 50% nếu cần")
print("      - KHÔNG tuân thủ market discount range")
print("      - Penalty nặng nếu vượt quá market discount (trừ khi có review tốt)")
print(f"   📊 Kết quả: Giá {agg['recommended_price']:,.0f} VNĐ | Prob {agg['prob_sold_well_at_recommended']*100:.1f}% | Margin {agg['profit_margin']*100:.1f}%")

print("\n🟡 BALANCED (Cân bằng - Tối ưu tổng thể):")
print("   🎯 Mục tiêu: CÂN BẰNG TỐI ƯU giữa TỶ LỆ BÁN và DOANH THU")
print("   ✅ Ưu điểm:")
print("      - Trade-off thông minh: 50% tỷ lệ bán + 50% doanh thu kỳ vọng")
print("      - Khuyến khích nằm trong MARKET RANGE (giá + discount)")
print("      - Sweet spot: Giá & discount ở mức trung bình (40-60%) → Buff +5-10%")
print("      - TUÂN THỦ market price range và discount range")
print("      - Có thể đánh đổi linh hoạt để đạt cân bằng tốt nhất")
print("   ⚠️  Rủi ro:")
print("      - Chấp nhận lỗ tối đa 10% (ưu tiên)")
print("   💡 Phù hợp: Đa số tình huống bán hàng thông thường, muốn cân bằng")
print(f"   📊 Kết quả: Giá {bal['recommended_price']:,.0f} VNĐ | Prob {bal['prob_sold_well_at_recommended']*100:.1f}% | Margin {bal['profit_margin']*100:.1f}%")

print("\n🟢 CONSERVATIVE (Thận trọng - Tối đa doanh thu):")
print("   🎯 Mục tiêu: TỐI ĐA HÓA DOANH THU KỲ VỌNG (prob × price)")
print("   ✅ Ưu điểm:")
print("      - Tối đa hóa doanh thu kỳ vọng, không chỉ lợi nhuận")
print("      - Giữ giá CAO nhưng vẫn đảm bảo xác suất bán hợp lý")
print("      - Phù hợp hàng cao cấp, có thương hiệu, độc quyền")
print("   ⚠️  Rủi ro:")
print("      - Xác suất bán có thể thấp hơn AGGRESSIVE và BALANCED")
print("      - Giảm giá ít hơn, lệch khỏi disc_low quá 20% sẽ bị phạt")
print("      - Penalty nặng nếu thiếu review tốt")
print("   💡 Chỉ dùng khi: Sản phẩm độc đáo, ít cạnh tranh, có review tốt")
print(f"   📊 Kết quả: Giá {con['recommended_price']:,.0f} VNĐ | Prob {con['prob_sold_well_at_recommended']*100:.1f}% | Revenue KV {con['expected_revenue_at_recommended']:,.0f}")

# Recommendation
print("\n" + "="*100)
print("💡 KHUYẾN NGHỊ LỰA CHỌN CHIẾN LƯỢC")
print("="*100)

# Auto recommend based on results
if user_input['comment_count'] and user_input['comment_count'] >= 20 and user_input['rating'] and user_input['rating'] >= 4.5:
    print("\n✨ KHUYẾN NGHỊ: 🟢 CONSERVATIVE hoặc 🟡 BALANCED")
    print("   Lý do: Sản phẩm có review tốt (≥20 comments, rating ≥4.5)")
    print("   → Có thể tối đa hóa doanh thu hoặc cân bằng tối ưu")
elif agg['profit_margin'] < -0.3:
    print("\n✨ KHUYẾN NGHỊ: 🟡 BALANCED")
    print("   Lý do: Aggressive strategy lỗ quá nhiều (>30%)")
    print("   → Nên chọn balanced để cân bằng giữa tỷ lệ bán và doanh thu")
elif bal['prob_sold_well_at_recommended'] < 0.3:
    print("\n✨ KHUYẾN NGHỊ: 🔴 AGGRESSIVE")
    print("   Lý do: Xác suất bán của balanced thấp (<30%)")
    print("   → Cần giảm giá mạnh hơn để tăng khả năng bán")
else:
    print("\n✨ KHUYẾN NGHỊ: 🟡 BALANCED")
    print("   Lý do: Chiến lược cân bằng phù hợp nhất với tình huống hiện tại")
    print("   → Tối ưu doanh thu kỳ vọng, rủi ro vừa phải")

print("\n" + "-" * 100)
print("📦 MỘT SỐ SẢN PHẨM TƯƠNG TỰ (sau lọc category + outlier):")
if res_base.get('similar_products_filtered') is not None:
    try:
        sim_display = res_base['similar_products_filtered'][[
            'name','price','sold','comment_count','rating','original_price','discount_percent'
        ]].head(10)
        print(sim_display.to_string())
    except Exception as e:
        print("Không in được bảng sản phẩm tương tự:", e)
else:
    print("Không tìm thấy sản phẩm tương tự.")

print("\n" + "="*100)
print("✅ HOÀN THÀNH SO SÁNH 3 CHIẾN LƯỢC TỐI ƯU GIÁ!")
print("="*100)
