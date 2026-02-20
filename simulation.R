# simulate labels for PAIR paper
# A and B labels that differ by x (called beta in paper)
# 4 final data sets: 
# balanced -- bal_ct labels per tweet, 50% A, 50% B
# unbalanced1 -- unbal1_ct labels per tweet, 67% A, 33% B
# unbalanced2 -- bal_ct labels per tweet, 75% A, 25% B
# adjusted -- bal_ct labels per tweet, 50% A, 50% B

require(tidyverse)
require(ggplot2)
require(janitor)
require(vtable)
require(skimr)
require(hash)
library("tidylog", warn.conflicts = FALSE)

set.seed(1210)

# number of labels per tweet in balanced data set
bal_ct <- 12

# number of labels per tweet in unbalanced 1 data set
unbal1_ct <- 9

# this parameter called beta in paper
xvalues <- expand.grid(x = seq(.05, .3, by=.05))


# gold dataset not mentioned in paper
# but used to create all other data sets
gold <- read_csv("data_Bolei/gold_train.csv") |>
  clean_names()  |>
  # drop these variables and remake after reducing from 15 to 12
  select(-p_i_hs, -p_i_ol) %>% 
  # drop obs with missing labels
  filter(!is.na(hate_speech) & !is.na(offensive_language)) %>% 
  rename(hs_gold = hate_speech, 
         ol_gold = offensive_language) %>% 
  # random sort -- because order used below
  mutate(rand = runif(nrow(.))) %>% 
  arrange(tweet_id, rand) %>% 
  group_by(tweet_id) |>
  # select bal_ct labels for each tweet (among non-missing labels)
  slice_sample(n=bal_ct) %>% 
  mutate(lnum = row_number(),
         tweet_label_id = row_number(),
         p_i_hs = mean(hs_gold, na.rm = TRUE),
         p_i_ol = mean(ol_gold, na.rm = TRUE)) |>
  ungroup()
tabyl(gold, lnum)
tabyl(gold, hs_gold)
tabyl(gold, ol_gold)
nrow(gold)
n_distinct(gold$tweet_id)
stopifnot(n_distinct(gold$tweet_id) * bal_ct == nrow(gold))

# create tweet-x level data set
tweet_x <- gold |>
  # reduce to tweet level
  select(tweet_id, tweet_hashed, p_i_hs, p_i_ol)  |>
  distinct_all() |>
  # cross with possible x values
  cross_join(xvalues) |>
  # generate pa and pb for this x value
  mutate(p_a_hs = pmax(p_i_hs - x, 0),
         p_a_ol = pmax(p_i_ol - x, 0),
         p_b_hs = pmin(p_i_hs + x, 1),
         p_b_ol = pmin(p_i_ol + x, 1))
stopifnot(n_distinct(tweet_x$tweet_id) == n_distinct(gold$tweet_id))

summary(tweet_x$p_i_hs)
summary(tweet_x$p_a_hs)
summary(tweet_x$p_b_hs)

summary(tweet_x$p_i_ol)
summary(tweet_x$p_a_ol)
summary(tweet_x$p_b_ol)


# create balanced data set
# bal_ct/2 A labels, bal_ct/2 B labels
bal <- tweet_x |>
  # add rows for bal_ct labels of each tweet
  cross_join(expand.grid(lnum = seq(1, bal_ct, 1))) |>
  rowwise() |>
  # first half of labels come from A, rest from B
  mutate(ol = if_else(lnum <= bal_ct/2,
                      rbinom(1,1,p_a_ol),
                      rbinom(1,1,p_b_ol)),
         hs = if_else(lnum <= bal_ct/2,
                      rbinom(1,1,p_a_hs),
                      rbinom(1,1,p_b_hs)),
         lab_source = if_else(lnum <= bal_ct/2,
                              "A",
                              "B")) |>
  ungroup()  |>
  group_by(tweet_id, x) |>
  mutate(tweet_label_id = row_number()) |>
  ungroup() |>
  arrange(tweet_id, tweet_label_id)

tabyl(bal, lnum)
tabyl(bal, lab_source)


# drop some B labels
unbal1 <- bal |>
  # keep subset of obs
  # because B labels have higher lnum values, will drop B labels
  filter(lnum <= unbal1_ct) |>
  group_by(tweet_id, x) |>
  mutate(tweet_label_id = row_number()) |>
  ungroup() |>
  arrange(tweet_id, x, tweet_label_id)
tabyl(unbal1, lnum)
tabyl(unbal1, x)


# add in more A labels to get back to bal_ct labels per tweet
unbal2 <- unbal1 |>
  bind_rows(tweets |>
              # number of dupe labels
              cross_join(expand.grid(extra_a_lab = seq(1, bal_ct - unbal1_ct, 1))) |>
              rowwise() |>
              # generate A labels for these cases
              mutate(ol = rbinom(1, 1, p_a_ol),
                     hs = rbinom(1, 1, p_a_hs),
                     lab_source = "A") |>
              ungroup()) |>
  group_by(tweet_id, x) |>
  mutate(tweet_label_id = row_number()) |>
  ungroup() |>
  arrange(tweet_id, x, tweet_label_id)
tabyl(unbal2, lnum)
tabyl(unbal2, extra_a_lab)
tabyl(unbal2, lnum, extra_a_lab)

adj <- unbal1  |>
  # append B tweets again
  bind_rows(unbal1 |> filter(lnum > bal_ct/2),
            .id = "dupe") |>
  group_by(tweet_id, x) |>
  mutate(tweet_label_id = row_number()) |>
  ungroup() |>
  arrange(tweet_id, tweet_label_id)
tabyl(adj, lnum)
tabyl(adj, dupe)
tabyl(adj, lnum, dupe)



## QC number of labels per tweet in each data set
bal_qc <- bal %>% 
  group_by(tweet_id,x) %>% 
  summarise(ct = n()) %>% 
  ungroup()
stopifnot(bal_qc$ct == bal_ct)

unbal1_qc <- unbal1 %>% 
  group_by(tweet_id,x) %>% 
  summarise(ct = n()) %>% 
  ungroup()
stopifnot(unbal1_qc$ct == unbal1_ct)

unbal2_qc <- unbal2 %>% 
  group_by(tweet_id,x) %>% 
  summarise(ct = n()) %>% 
  ungroup()
stopifnot(unbal2_qc$ct == bal_ct)

adj_qc <- adj %>% 
  group_by(tweet_id,x) %>% 
  summarise(ct = n()) %>% 
  ungroup()
stopifnot(adj_qc$ct == bal_ct)


# put all datasets together
all <- gold %>% 
  group_by(tweet_id) %>% 
  slice_sample(n=bal_ct) %>% 
  ungroup() %>% 
  mutate(dataset = "gold") %>% 
  rename(hs = hs_gold, 
         ol = ol_gold) %>% 
  bind_rows(bal %>% 
              mutate(dataset = "bal")) %>% 
  bind_rows(unbal1 %>% 
              mutate(dataset = "unbal1")) %>% 
  bind_rows(unbal2 %>% 
              mutate(dataset = "unbal2")) %>% 
  bind_rows(adj %>% 
              mutate(dataset = "adj")) %>% 
  mutate(rowid = row_number()) 

tabyl(all, dataset)
tabyl(all, dataset, lab_source)

# data sets for model training
write_csv(all %>% 
            select(-extra_a_lab, -dupe, -lnum, -id, -version, -batch_tweet, -rand), "all_labels.csv")

# save out individual data sets
saveRDS(all, "all.rds")
saveRDS(bal, "bal.rds")
saveRDS(unbal1, "unbal1.rds")
saveRDS(unbal2, "unbal2.rds")
saveRDS(adj, 'adj.rds')
