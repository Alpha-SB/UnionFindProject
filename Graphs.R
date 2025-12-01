library(tidyverse)
library(ggplot2)

df <- read_csv("results.csv")
glimpse(df)

# runtime vs n (50-50 workload)
df_5050 <- df %>% 
  filter(workload == "random_50_50")

ggplot(df_5050, aes(x = n, y = runtime, color = structure)) +
  geom_point() +
  geom_line() +
  labs(
    title = "Runtime vs n (50/50 Workload)",
    x = "n",
    y = "Runtime (seconds)",
    color = "UF Structure"
  ) +
  theme_minimal()

# runtime vs n (all Workloads)
ggplot(df, aes(x = n, y = runtime, color = structure)) +
  geom_point() +
  geom_line() +
  facet_wrap(~ workload, scales = "free_y") +
  labs(
    title = "Runtime vs n Across All Workloads",
    x = "n",
    y = "Runtime (seconds)"
  ) +
  theme_minimal()


# pointer updates vs n (adversarial chain)
df_adv <- df %>% 
  filter(workload == "adversarial")

ggplot(df_adv, aes(x = n, y = pointer_updates, color = structure)) +
  geom_point() +
  geom_line() +
  labs(
    title = "Pointer Updates vs n (Adversarial Workload)",
    x = "n",
    y = "Pointer Updates",
    color = "UF Structure"
  ) +
  theme_minimal()

# general pointer updates (tree vs list)
ggplot(df, aes(x = n, y = pointer_updates, color = structure)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(
    title = "Pointer Updates Across All Workloads",
    x = "n",
    y = "Pointer Updates",
    color = "UF Structure"
  ) +
  theme_minimal()


unique(df$workload)
df %>% filter(workload == "adversarial") %>% count(structure)
df %>% filter(workload == "adversarial") %>% select(n, structure, pointer_updates) %>% arrange(n)

