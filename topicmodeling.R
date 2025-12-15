# ===== R Topic Modeling for A01–A89 (with printed output) =====
library(tidyverse)
library(tidytext)
library(readr)
library(stringr)
library(topicmodels)
library(tm)
library(dplyr)
library(tidyr)
library(ggrepel)
library(stringi)   # robust unicode normalization
library(Matrix)    # sparse matrices & ops
if (!requireNamespace("wordcloud", quietly = TRUE)) install.packages("wordcloud")
if (!requireNamespace("RColorBrewer", quietly = TRUE)) install.packages("RColorBrewer")
library(wordcloud)
library(RColorBrewer)

# --- Paths ---
corpus_dir <- "C:/Users/A.Yakout/Desktop/MASTERS/BANA655/Project/Research_txt - NUMBERS ONLY"
out_dir    <- "C:/Users/A.Yakout/Desktop/MASTERS/BANA655/Project/helphelp"; dir.create(out_dir, FALSE, TRUE)

# --- Load files ---
files <- list.files(corpus_dir, "^A\\d+\\.txt$", full.names = TRUE)
stopifnot(length(files) > 0)
cat("✅ Loaded", length(files), "text files\n")

docs <- tibble(
  doc_id = str_match(basename(files), "^(A\\d+)\\.txt$")[,2],
  text   = map_chr(files, read_file)
) |> arrange(doc_id)

# --- Tokenize & clean (normalize; drop apostrophes; map U.S./USA; keep a–z only) ---
tokens <- docs |>
  unnest_tokens(word, text, token = "words", to_lower = TRUE, strip_punct = FALSE) |>
  mutate(
    word = stringi::stri_trans_general(word, "NFKC"),
    word = str_replace_all(word, "['’`´]", ""),
    word = str_replace_all(word, "(?i)\\bu\\.s\\.a?\\.?\\b", "united_states"),
    word = str_replace_all(word, "[^a-z]", "")
  ) |>
  filter(word != "") |>
  anti_join(stop_words, by = "word")

# Custom stops (remove 'its', 'united_states', web cruft, common fillers)
custom_stop <- tibble(
  word = c("https","http","www","com","amp","said",
           "use","one","two","like","also","eg","e.g",
           "its","united_states")
)
tokens <- tokens |> anti_join(custom_stop, by = "word")

cat("✅ Tokenization complete:", nrow(tokens), "tokens total\n")

# --- Document-Term Matrix ---
dtm <- tokens |> count(doc_id, word, name = "n") |>
  cast_dtm(document = doc_id, term = word, value = n)
dtm <- dtm[slam::row_sums(dtm) > 0, ]
cat("✅ DTM built with", nrow(dtm), "documents and", ncol(dtm), "terms\n")

# --- Save DTM as Matrix Market + indices ---
dtm_sparse <- Matrix::Matrix(as.matrix(dtm), sparse = TRUE)
Matrix::writeMM(dtm_sparse, file.path(out_dir, "dtm.mtx"))
write_csv(tibble(doc_id = rownames(dtm)), file.path(out_dir, "dtm_docs.csv"))
write_csv(tibble(term = colnames(dtm)), file.path(out_dir, "dtm_terms.csv"))
cat("✅ DTM saved as Matrix Market (dtm.mtx + dtm_docs.csv + dtm_terms.csv)\n")

# --- Top-100 words (same cleaning)
word_freq <- tokens |> count(word, sort = TRUE)
top100 <- head(word_freq, 100)
write_csv(top100, file.path(out_dir, "top100_words.csv"))
cat("✅ Exported top100_words.csv\n")

# --- DTM restricted to Top 100 terms ---
top_terms <- unique(top100$word)
keep_cols <- intersect(colnames(dtm), top_terms)
dtm_top100 <- dtm[, keep_cols, drop = FALSE]
dtm_top100 <- dtm_top100[slam::row_sums(dtm_top100) > 0, ]

dtm_top100_dense <- as.matrix(dtm_top100)
write.csv(dtm_top100_dense, file.path(out_dir, "dtm_top100.csv"), row.names = TRUE)
saveRDS(dtm_top100, file.path(out_dir, "dtm_top100.rds"))
cat("✅ DTM (Top 100 terms) saved as CSV and RDS\n")

# --- PCA on TF-IDF (Top-100 terms) ---
tf_all <- slam::row_sums(dtm_top100)
df_all <- slam::col_sums(dtm_top100 > 0)
N_docs <- nrow(dtm_top100)
idf    <- log((N_docs + 1) / (df_all + 1)) + 1
TF <- dtm_top100
rs <- pmax(tf_all, 1)              # avoid division by 0
TF$v <- TF$v / rs[TF$i]            # row-normalized TF
TFIDF <- TF
TFIDF$v <- TFIDF$v * idf[TFIDF$j]  # multiply by IDF
X_top <- as.matrix(TFIDF)
pca_top <- prcomp(X_top, center = TRUE, scale. = TRUE)
coords_top <- tibble::as_tibble(pca_top$x[, 1:2], rownames = "doc_id")

p_pca100 <- ggplot(coords_top, aes(PC1, PC2, label = doc_id)) +
  geom_point() +
  ggrepel::geom_text_repel(size = 3, max.overlaps = 30) +
  labs(title = "Documents in Top-100 TF-IDF Space (PCA)")
ggsave(file.path(out_dir, "viz_dtm_top100_pca.png"), p_pca100, width = 8, height = 6, dpi = 180)
cat("✅ Saved viz_dtm_top100_pca.png\n")

# --- Top-30 bar chart ---
p_top <- top100 %>%
  slice_max(n, n = 30) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 30 Most Frequent Words", x = NULL, y = "Count")
ggsave(file.path(out_dir, "viz_top30_words.png"), p_top, width = 8, height = 6, dpi = 180)
cat("✅ Saved viz_top30_words.png\n")

# --- Word Cloud ---
png(file.path(out_dir, "viz_top100_wordcloud.png"), width = 900, height = 650)
set.seed(123)
wordcloud(
  words = top100$word,
  freq  = top100$n,
  scale = c(4, 0.6),
  min.freq = 1,
  colors = brewer.pal(8, "Dark2")
)
dev.off()
cat("✅ Saved viz_top100_wordcloud.png\n")

# --- LDA Topic Modeling ---
set.seed(0)
K <- 12
lda <- LDA(dtm, k = K, method = "Gibbs",
           control = list(seed = 0, iter = 1500, alpha = 0.2))
cat("✅ LDA completed with", K, "topics\n\n")

# --- Extract results ---
beta  <- tidytext::tidy(lda, matrix = "beta")   # topic-term weights
gamma <- tidytext::tidy(lda, matrix = "gamma")  # doc-topic weights

# --- Topic summary ---
top_terms_tbl <- beta |>
  group_by(topic) |>
  slice_max(beta, n = 10, with_ties = FALSE) |>
  summarise(top_terms = paste(term, collapse = ", "), .groups = "drop")

topic_prev <- gamma |>
  group_by(topic) |>
  summarise(prevalence = mean(gamma), .groups = "drop") |>
  arrange(desc(prevalence))

topic_summary <- left_join(topic_prev, top_terms_tbl, by = "topic")

# Print summary
cat("=== Topic Summary ===\n")
print(topic_summary, n = K)
cat("\n")

# --- Per-document topic assignments ---
doc_topic_long <- gamma |>
  rename(doc_id = document, probability = gamma) |>
  arrange(doc_id, desc(probability))

dominant <- doc_topic_long |>
  group_by(doc_id) |>
  slice_max(probability, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(doc_id,
            dominant_topic = paste0("Topic_", topic),
            max_prob = round(probability, 3))

# Print first few dominant topics
cat("=== Dominant Topic per Document (first 10) ===\n")
print(head(dominant, 10))
cat("\n")

# --- Save outputs ---
write_csv(topic_summary, file.path(out_dir, "topic_summary.csv"))
write_csv(doc_topic_long, file.path(out_dir, "per_document_topic_assignments_long.csv"))
write_csv(
  doc_topic_long |> pivot_wider(names_from = topic, values_from = probability, values_fill = 0),
  file.path(out_dir, "per_document_topic_assignments_wide.csv")
)
cat("✅ CSV files saved in:", out_dir, "\n")

# ---- Global white background theme ----
theme_set(
  theme_minimal(base_size = 11) +
    theme(plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA))
)

# ---- Visualizations ----
topic_summary  <- read_csv(file.path(out_dir, "topic_summary.csv"), show_col_types = FALSE)
doc_topic_long <- read_csv(file.path(out_dir, "per_document_topic_assignments_long.csv"), show_col_types = FALSE)
doc_topic_wide <- read_csv(file.path(out_dir, "per_document_topic_assignments_wide.csv"), show_col_types = FALSE)

# Normalize column names
if (!"doc_id" %in% names(doc_topic_wide)) {
  names(doc_topic_wide)[1] <- "doc_id"
}
if (!any(grepl("^Topic_", names(doc_topic_wide)))) {
  non_id <- setdiff(names(doc_topic_wide), "doc_id")
  names(doc_topic_wide)[names(doc_topic_wide) %in% non_id] <- paste0("Topic_", non_id)
}
topic_cols <- grep("^Topic_", names(doc_topic_wide), value = TRUE)

# Dominant topic per document (for colouring)
doc_topic_dom <- doc_topic_wide %>%
  select(doc_id, all_of(topic_cols)) %>%
  pivot_longer(-doc_id, names_to = "topic", values_to = "prob") %>%
  group_by(doc_id) %>%
  slice_max(prob, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(doc_id, dominant_topic = topic)

# (a) Topic prevalence bar chart
p_prev <- topic_summary %>%
  mutate(topic_lab = paste0("Topic ", topic)) %>%
  ggplot(aes(x = forcats::fct_reorder(topic_lab, prevalence), y = prevalence)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Topic Prevalence", x = NULL, y = "Share of corpus")
print(p_prev)
ggsave(file.path(out_dir, "viz_topic_prevalence_white.png"), p_prev, width = 7, height = 5, dpi = 180)

# (b) Top terms per topic (facets)
top_terms_plot <- beta %>%
  group_by(topic) %>%
  slice_max(beta, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(term = tidytext::reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta)) +
  geom_col(fill = "gray40") +
  coord_flip() +
  tidytext::scale_x_reordered() +
  facet_wrap(~ topic, scales = "free_y") +
  labs(title = "Top Terms per Topic", x = NULL, y = "β (term weight)")
print(top_terms_plot)
ggsave(file.path(out_dir, "viz_top_terms_per_topic_white.png"), top_terms_plot, width = 10, height = 7, dpi = 180)

# (c) Document × Topic heatmap
hm_df <- doc_topic_wide %>%
  select(doc_id, all_of(topic_cols)) %>%
  pivot_longer(-doc_id, names_to = "topic", values_to = "prob")
p_hm <- ggplot(hm_df, aes(x = topic, y = doc_id, fill = prob)) +
  geom_tile() +
  scale_fill_viridis_c(option = "C") +
  labs(title = "Document × Topic Heatmap", x = "Topic", y = "Document", fill = "Probability") +
  theme(axis.text.y = element_text(size = 6))
print(p_hm)
ggsave(file.path(out_dir, "viz_doc_topic_heatmap_white.png"), p_hm, width = 9, height = 11, dpi = 180)

# (d) Stacked bar per document (topic mixtures)
stack_df <- doc_topic_long %>%
  mutate(topic = paste0("Topic_", topic))
p_stack <- ggplot(stack_df, aes(x = doc_id, y = probability, fill = topic)) +
  geom_bar(stat = "identity") +
  labs(title = "Topic Mixture per Document (Stacked)", x = "Document", y = "Probability") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 6))
print(p_stack)
ggsave(file.path(out_dir, "viz_doc_topic_stacked_white.png"), p_stack, width = 12, height = 6, dpi = 180)

# (e) PCA: documents in topic space
X <- doc_topic_wide %>%
  select(all_of(topic_cols)) %>%
  as.matrix()
rownames(X) <- doc_topic_wide$doc_id
pca_docs <- prcomp(X, center = TRUE, scale. = TRUE)
coords_docs <- tibble::as_tibble(pca_docs$x[, 1:2], rownames = "doc_id") %>%
  left_join(doc_topic_dom, by = "doc_id")
p_pca <- ggplot(coords_docs, aes(PC1, PC2, color = dominant_topic, label = doc_id)) +
  geom_point(size = 2) +
  ggrepel::geom_text_repel(size = 3, max.overlaps = 20) +
  labs(title = "Documents in Topic Space (PCA)", color = "Dominant Topic")
print(p_pca)
ggsave(file.path(out_dir, "viz_doc_pca_white.png"), p_pca, width = 8, height = 6, dpi = 180)

# ===== (f) TF-IDF + Terms×Terms PCA & MDS (TOP 100 TERMS) ===================
# Build TF-IDF (terms × docs) from the already-cleaned tokens
tfidf_long <- tokens |>
  count(doc_id, word, name = "n") |>
  tidytext::bind_tf_idf(term = word, document = doc_id, n = n)

TFIDF_td <- tidytext::cast_sparse(row = word, column = doc_id, value = tf_idf)  # dgCMatrix terms×docs

# Select TOP 100 terms by total TF-IDF
max_terms <- 100
term_strength <- Matrix::rowSums(TFIDF_td)
keep_terms <- names(sort(term_strength, decreasing = TRUE))[seq_len(min(max_terms, length(term_strength)))]
TFIDF_sub <- TFIDF_td[keep_terms, , drop = FALSE]

# Color mapping: LDA dominant topic per term (from 'beta')
term_dom_topic <- beta |>
  group_by(term) |>
  slice_max(beta, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(term, dominant_topic = paste0("Topic_", topic))

# PCA (terms in document space via TF-IDF)
pca_terms <- prcomp(as.matrix(TFIDF_sub), center = TRUE, scale. = TRUE)
terms_pca_coords <- tibble::tibble(
  term = rownames(pca_terms$x),
  PC1  = pca_terms$x[, 1],
  PC2  = pca_terms$x[, 2]
) |>
  left_join(term_dom_topic, by = "term")

p_terms_pca <- ggplot(terms_pca_coords, aes(PC1, PC2, color = dominant_topic, label = term)) +
  geom_point(size = 1.8) +
  ggrepel::geom_text_repel(size = 3, max.overlaps = 30) +
  labs(title = "Terms × Terms (PCA on TF-IDF) — Top 100 Terms", color = "Dominant Topic")
print(p_terms_pca)
ggsave(file.path(out_dir, "viz_terms_pca_tfidf_top100_white.png"), p_terms_pca, width = 9, height = 7, dpi = 180)
readr::write_csv(terms_pca_coords, file.path(out_dir, "terms_pca_coords_top100.csv"))

# Quick check for a specific term (example: 'ai' if present)
ai_df <- tfidf_long |> filter(word == "ai") |> summarise(
  docs_with_ai = n_distinct(doc_id),
  mean_tfidf   = mean(tf_idf),
  sd_tfidf     = sd(tf_idf)
)
print(ai_df)

# Classical MDS from cosine distances between terms
row_norms <- sqrt(Matrix::rowSums(TFIDF_sub^2))
row_norms[row_norms == 0] <- 1
TFIDF_norm <- TFIDF_sub / row_norms

cos_sim <- as.matrix(TFIDF_norm %*% Matrix::t(TFIDF_norm))
cos_sim[is.na(cos_sim)] <- 0
cos_sim <- pmin(pmax(cos_sim, -1), 1)  # clamp to [-1,1]
cos_dist <- as.dist(1 - cos_sim)

mds_fit <- cmdscale(cos_dist, k = 2, eig = TRUE)
terms_mds_coords <- tibble::tibble(
  term = rownames(cos_sim),
  Dim1 = mds_fit$points[, 1],
  Dim2 = mds_fit$points[, 2]
) |>
  left_join(term_dom_topic, by = "term")

p_terms_mds <- ggplot(terms_mds_coords, aes(Dim1, Dim2, color = dominant_topic, label = term)) +
  geom_point(size = 1.8) +
  ggrepel::geom_text_repel(size = 3, max.overlaps = 30) +
  labs(title = "Terms × Terms (MDS from Cosine) — Top 100 Terms", color = "Dominant Topic")
print(p_terms_mds)
ggsave(file.path(out_dir, "viz_terms_mds_cosine_top100_white.png"), p_terms_mds, width = 9, height = 7, dpi = 180)
readr::write_csv(terms_mds_coords, file.path(out_dir, "terms_mds_coords_top100.csv"))
# ============================================================================

cat("✅ All plots saved (white background) in:", out_dir, "\n")
