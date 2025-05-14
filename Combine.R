# Practicum R File
# NFL Combine Data

# Load Data / Libraries
library(dplyr)
library(ggplot2)
library(car)
library(readxl)
library(tidyr)       
library(janitor)     
library(stringr)     
library(skimr)       
library(caret)     
library(tidymodels)  
library(lubridate)  
library(corrplot)  
library(ggcorrplot) 
library(ranger)
library(tibble)

# 2000 - 2025 NFL Combine Data
  # 2025 does not have draft data
  # 2021 is ONLY pro day data - combine cancelled bc of covid
combine <- read_excel('/Users/spencer/Desktop/Everything/School Documents/MSBA/Practicum/combine_data.xlsx')
summary(combine)

combine_clean <- combine %>%
  filter(!(is.na(`40yd`) & 
             is.na(Vertical) & 
             is.na(Bench) & 
             is.na(`Broad Jump`) & 
             is.na(`3Cone`) & 
             is.na(Shuttle)))


# Random Forest 40 Predictive Model - Take out obs without 40 time
combine_40 <- combine_clean %>%
  filter(!is.na(`40yd`))


combine40grouped <- combine_40 %>%
  mutate(PosGroup = case_when(
    Pos %in% c("OT", "OG", "C", "OL","G")       ~ "Offensive Line",
    Pos %in% c("DT", "DE", "DL")                ~ "Defensive Line",
    Pos %in% c("WR", "RB", "FB", "TE")          ~ "Skill (Offense)",
    Pos %in% c("CB", "S", "DB", "SAF")          ~ "Defensive Backs",
    Pos %in% c("QB")                            ~ "Quarterback",
    Pos %in% c("LB", "ILB", "OLB", "EDGE")      ~ "Linebacker",
    Pos %in% c("K", "P", "LS")                  ~ "Special Teams",
    TRUE                                        ~ "Other"
  ))


# Split into training / testing (80% / 20%)
set.seed(5)  # so it’s reproducible
split_40 <- initial_split(combine_40, prop = 0.8)
train_40 <- training(split_40)
test_40 <- testing(split_40)


# Create a recipe for preprocessing to handle NAs, dummy variables, etc.
recipe_40 <- recipe(`40yd` ~ Height + Weight + Vertical + Bench +
                      `Broad Jump` + `3Cone` + Shuttle + Pos,
                    data = train_40) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())


# Specify the random forest model to define model + tuning 
  # mtry and min_n control tree structure and need tuning
rf_spec <- rand_forest(
  mtry = tune(),         # # of variables to consider at each split
  min_n = tune(),        # min # of obs in a node before splitting
  trees = 500            # total trees in the forest
) %>%
  set_engine("ranger") %>%
  set_mode("regression")




# Tune model via cross validation
  # try different combinations of mtry and min_n to find the best-performing model
# Create 5-fold cross-validation object
set.seed(5)
cv_folds <- vfold_cv(train_40, v = 5)

# Run tuning grid search
# Reduce the number of trees to 100 (less intensive but still valid)
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 100
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")


# Combine recipe and model in workflow
rf_wf <- workflow() %>%
  add_recipe(recipe_40) %>%
  add_model(rf_spec)


# Use a smaller tuning grid (10 combos instead of 20)
rf_results <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid = 10,  # Less parameter combos = faster
  metrics = metric_set(rmse, rsq, mae)
)

show_best(rf_results, metric = "rmse")
#What it does:
  #Lists the top-performing combinations of mtry and min_n, 
  #sorted by RMSE (Root Mean Squared Error)
# Why it matters:
  # RMSE is like your model’s 40-yard dash time: lower = better
  # This tells you which configuration performed best across cross-validation folds

autoplot(rf_results)
# What it does:
  # Plots RMSE (or another metric) against tuning parameters
# Why it matters:
  # You can visually spot trends, like:
  # “RMSE gets better as mtry increases”
  # “Performance flattens out at certain parameter values”
  # Helps you avoid overfitting by picking a model that performs consistently well


best_rf <- select_best(rf_results, metric="rmse")
# What it does:
  # From all the tuning combos you tried (in tune_grid()), 
  # this pulls out the one with the lowest RMSE.
# Why it matters:
  # You want to finalize your model using the best-performing hyperparameters.
  # Think of this like saying: “Of all the ways I grew this forest, 
  # which combination of mtry and min_n gave me the most accurate predictions?”

final_rf <- finalize_workflow(rf_wf, best_rf)
# What it does:
  # Takes your original workflow (recipe + random forest model spec)
  # Plugs in the winning hyperparameters you just selected
  # Outputs a ready-to-train model with everything set
# Why it matters:
  # You now have a finalized workflow you can train on the full training set.

final_fit <- fit(final_rf, data = train_40)
# What it does:
  # Trains your finalized model using all of your training data
# Why now?
  # During tuning, the model was trained in chunks (folds).
  # Now that we’ve picked the best configuration, we retrain it once on all of the 
  # training data, so it learns as much as possible before facing the test set.


# Predictions
rf_preds <- predict(final_fit, new_data = test_40) %>%
  bind_cols(test_40 %>% select(`40yd`))
# What it does:
  # Predicts 40-yard dash times for players in the test set and combines predictions
  # with their actual 40yd times.
# Why:
  # You need both the prediction and the truth side-by-side to calculate how 
  # well the model works.


# Evaluate model
metrics(rf_preds, truth = `40yd`, estimate = .pred)
# What it shows:
  # RMSE: Average prediction error (sensitive to big misses)
  # MAE: Average absolute error (easy to interpret, like “off by 0.10 seconds”)
  # R^2: How much of the variation in 40 times your model explains (closer to 1 is better)


# Which variables had the most impact
rf_model <- final_fit %>% extract_fit_parsnip()
sort(rf_model$fit$variable.importance, decreasing = TRUE)


# Turn the importance vector into a tidy dataframe
importance_df <- rf_model$fit$variable.importance %>%
  sort(decreasing = TRUE) %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
  rename(Importance = ".")

# Export to CSV for use in Tableau
write.csv(importance_df, "/Users/spencer/Downloads/variable_importance_rf.csv", row.names = FALSE)


# Plot the top 10 most important variables
ggplot(importance_df[1:10, ], aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Most Important Predictors of 40-Yard Dash Time",
       x = "Variable",
       y = "Importance (Impurity Reduction)") +
  theme_minimal()




# Predicted vs Actual Scatterplot
ggplot(rf_preds, aes(x = `40yd`, y = .pred)) +
  geom_point(alpha = 0.6) +
  geom_abline(linetype = "dashed", color = "red") +
  labs(title = "Actual vs Predicted 40-Yard Times",
       x = "Actual 40 Time",
       y = "Predicted 40 Time") +
  theme_minimal()


# Residual plot - where is model off & by how much?
rf_preds %>%
  mutate(residual = .pred - `40yd`) %>%
  ggplot(aes(x = .pred, y = residual)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Predicted 40-Yard Times",
       x = "Predicted 40 Time",
       y = "Residual (Predicted - Actual)") +
  theme_minimal()


# Density Plot of Predicted vs Actual
rf_preds %>%
  pivot_longer(cols = c(`40yd`, .pred), names_to = "type", values_to = "time") %>%
  ggplot(aes(x = time, fill = type)) +
  geom_density(alpha = 0.4) +
  labs(title = "Distribution of Actual vs Predicted 40 Times",
       x = "40 Time (seconds)", fill = "") +
  theme_minimal()




# Export actual & predicted to visualize in tableau
# Add PosGroup to the test set
test_40_grouped <- test_40 %>%
  mutate(PosGroup = case_when(
    Pos %in% c("OT", "OG", "C", "OL","G")       ~ "Offensive Line",
    Pos %in% c("DT", "DE", "DL")                ~ "Defensive Line",
    Pos %in% c("WR", "RB", "FB", "TE")          ~ "Skill (Offense)",
    Pos %in% c("CB", "S", "DB", "SAF")          ~ "Defensive Backs",
    Pos %in% c("QB")                            ~ "Quarterback",
    Pos %in% c("LB", "ILB", "OLB", "EDGE")      ~ "Linebacker",
    Pos %in% c("K", "P", "LS")                  ~ "Special Teams",
    TRUE                                        ~ "Other"
  ))

# Merge PosGroup and Pos into the predictions
rf_preds3 <- predict(final_fit, new_data = test_40_grouped) %>%
  bind_cols(test_40_grouped %>% select(`40yd`, Pos, PosGroup))

# Prepare and export to CSV
rf_preds3_export <- rf_preds3 %>%
  mutate(PlayerID = row_number()) %>%
  rename(Actual = `40yd`,
         Predicted = .pred) %>%
  select(PlayerID, Actual, Predicted, Pos, PosGroup)


write.csv(rf_preds3_export, "/Users/spencer/Downloads/rf_40_predictions4.csv", row.names = FALSE)


combine_clean2 <- combine_clean %>%
  mutate(PosGroup = case_when(
    Pos %in% c("OT", "OG", "C", "OL","G")       ~ "Offensive Line",
    Pos %in% c("DT", "DE", "DL")                ~ "Defensive Line",
    Pos %in% c("WR", "RB", "FB", "TE")          ~ "Skill (Offense)",
    Pos %in% c("CB", "S", "DB", "SAF")          ~ "Defensive Backs",
    Pos %in% c("QB")                            ~ "Quarterback",
    Pos %in% c("LB", "ILB", "OLB", "EDGE")      ~ "Linebacker",
    Pos %in% c("K", "P", "LS")                  ~ "Special Teams",
    TRUE                                        ~ "Other"
  ))

write.csv(combine_clean2, "/Users/spencer/Downloads/combine_clean2.csv", row.names = FALSE)




# Select numeric columns from your combine_clean2 dataset
cor_data2 <- combine_clean2 %>%
  select(where(is.numeric)) %>%
  drop_na()

# Compute correlation matrix
cor_matrix2 <- cor(cor_data, use = "complete.obs")

# Plot heatmap
ggcorrplot(cor_matrix,
           method = "circle",
           type = "lower",
           lab = TRUE,
           lab_size = 3,
           colors = c("red", "white", "blue"),
           title = "Correlation Heatmap of Combine Metrics",
           ggtheme = theme_minimal())




# Load libraries (if not already loaded)
library(dplyr)
library(janitor)
library(scales)  # for rescale()

combinegrouped <- combine %>%
  mutate(PosGroup = case_when(
    Pos %in% c("OT", "OG", "C", "OL","G")       ~ "Offensive Line",
    Pos %in% c("DT", "DE", "DL")                ~ "Defensive Line",
    Pos %in% c("WR", "RB", "FB", "TE")          ~ "Skill (Offense)",
    Pos %in% c("CB", "S", "DB", "SAF")          ~ "Defensive Backs",
    Pos %in% c("QB")                            ~ "Quarterback",
    Pos %in% c("LB", "ILB", "OLB", "EDGE")      ~ "Linebacker",
    Pos %in% c("K", "P", "LS")                  ~ "Special Teams",
    TRUE                                        ~ "Other"
  ))


# Select numeric drill columns + position group
drill_avgs <- combinegrouped %>%
  select(PosGroup, `40yd`, Bench, `Broad Jump`, Shuttle, `3Cone`, Vertical) %>%
  drop_na() %>%  # Drop rows with any missing drill data
  group_by(PosGroup) %>%
  summarise(
    `40yd` = mean(`40yd`, na.rm = TRUE),
    Bench = mean(Bench, na.rm = TRUE),
    `Broad Jump` = mean(`Broad Jump`, na.rm = TRUE),
    Shuttle = mean(Shuttle, na.rm = TRUE),
    `3Cone` = mean(`3Cone`, na.rm = TRUE),
    Vertical = mean(Vertical, na.rm = TRUE)
  ) %>%
  ungroup()

# Normalize each column for radar chart (e.g., min-max scaling)
#  ensures fair comparison across drills
drill_avgs_scaled <- drill_avgs %>%
  mutate(
    `40yd` = rescale(-`40yd`),  # Flip 40 time so lower = better
    Bench = rescale(Bench),
    `Broad Jump` = rescale(`Broad Jump`),
    Shuttle = rescale(-Shuttle),     # Lower = better for agility
    `3Cone` = rescale(-`3Cone`),     # Lower = better for agility
    Vertical = rescale(Vertical)
  )

# Pivot longer for Tableau format
library(tidyr)

drill_long <- drill_avgs_scaled %>%
  pivot_longer(
    cols = -PosGroup,
    names_to = "Drill",
    values_to = "Avg_Score"
  )

drill_long <- drill_long %>%
  mutate(Drill_Order = case_when(
    Drill == "40yd" ~ 1,
    Drill == "Shuttle" ~ 2,
    Drill == "3Cone" ~ 3,
    Drill == "Bench" ~ 4,
    Drill == "Broad Jump" ~ 5,
    Drill == "Vertical" ~ 6
  ))

# Export to CSV
write.csv(drill_long, "/Users/spencer/Downloads/radar_drill_scores.csv", row.names = FALSE)




library(dplyr)

# Define the drill columns
drills <- c("40yd", "Bench", "Broad Jump", "Shuttle", "3Cone", "Vertical")

# Flip direction for drills where lower = better
combine_flipped <- combine %>%
  mutate(across(all_of(drills), as.numeric)) %>%
  mutate(
    `40yd` = -`40yd`,
    Shuttle = -Shuttle,
    `3Cone` = -`3Cone`
  )

# Calculate 90th percentile cutoffs for each drill
elite_cutoffs <- combine_flipped %>%
  summarise(across(all_of(drills), ~ quantile(., 0.95, na.rm = TRUE)))

# Flag if each player exceeds the elite threshold per drill
elite_flags <- combine_flipped %>%
  mutate(across(all_of(drills), ~ . > elite_cutoffs[[cur_column()]], .names = "elite_{col}")) %>%
  rowwise() %>%
  mutate(elite_count = sum(c_across(starts_with("elite_")), na.rm = TRUE)) %>%
  ungroup()

# Filter to show only players with 3 or more elite performances
combine_elite <- elite_flags %>%
  filter(elite_count >= 4) %>%
  select(Player, Pos, School, elite_count, starts_with("elite_"))

# View or export
View(combine_elite)
