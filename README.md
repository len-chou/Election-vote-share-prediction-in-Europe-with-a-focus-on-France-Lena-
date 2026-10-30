# Election vote share prediction

**Can we predict, before an election, what share of the vote each party will get?**

This project trains machine learning models on 30 years of European national parliament elections to predict each party's vote share. It then uses them for one real future election: **France 2027**, compared with the latest polls.

It is a regression problem. The target is `vote_share`, the % of votes a party gets in one election. Every feature is something that was known **before** election day, so the model never sees the result it is trying to predict.

---

## Key findings

| | Test error 2015–2023 (MAE, points) |
|---|---|
| Baseline: "same score as last time" | 3.59 |
| Decision Tree (tuned) | 3.55 |
| Bagging | 3.46 |
| Random Forest | 3.42 |
| AdaBoost | 3.42 |
| **Gradient Boosting (final model)** | **3.15** |

- **Gradient Boosting is the best model.** It is off by 3.15 points on average on elections it never saw (2015–2023), which is **12% better than the baseline**. Its R² is 0.75, against 0.68 for the baseline.
- **8 times out of 10, the model is within 5 points** of the real result (81%, against 76% for the baseline).
- **A party's last result is by far the strongest clue.** After that come being in government (parties in power tend to lose votes) and two features built for this project: how many new parties are running, and how much of the last vote is "up for grabs". The economy adds very little on its own.
- **What the model cannot do:** it does not pick the winner better than the baseline (64% vs 65%). It cannot see the big shocks that change who comes first, such as a new movement, a collapse, or an alliance. Most of its gain comes from small and new parties.
- **Alliances break party-level models.** In France 2022, the NUPES left alliance was stored under one party name, so the party model predicted only 13.6% for the left when it really got 28.3%. Predicting **political blocs** instead of parties fixes this: the bloc model predicted 25.2%.

### France 2027: final estimate by bloc

The bloc model (based on history) is blended 50/50 with the 2027 presidential polls, which are first translated into parliament-election terms using the gap observed in 2022.

| Bloc | Parliament vote | Presidential 1st round |
|---|---|---|
| Far right | 31.8% | 39.1% |
| Centre | 16.9% | 19.0% |
| Left | 16.5% | 12.7% |
| Far left | 12.9% | 20.4% |
| Right | 11.8% | 7.3% |

This shows how strong each bloc is, not who will win. The presidential election is decided in the second round, which this project does not model.

---

## Data

| Source | What it gives | Access |
|---|---|---|
| [ParlGov](https://www.parlgov.org/) (`view_election.csv`, `view_party.csv`, `view_cabinet.csv`) | Every party's result in every national election, its political family and left–right position, and the governments in office | Free download (Data → CSV) |
| [World Bank Open Data](https://data.worldbank.org/) | GDP growth (`NY.GDP.MKTP.KD.ZG`), unemployment (`SL.UEM.TOTL.ZS`), inflation (`FP.CPI.TOTL.ZG`) | Free download (CSV) |
| Sondax: 2027 French presidential polls (Kaggle) | 46 polls from 10 institutes (Feb–Oct 2026), one row per candidate per scenario | `data/sondages-presidentielle-2027.csv` |
| French Ministry of the Interior | 2024 snap legislative election results, typed in by hand because ParlGov stops in 2022 for France | Values written in notebooks 04 and 05 |

**Final table (`data/party_elections.csv`):** 2,575 rows, 271 national parliament elections, 31 European countries, 1991–2023. One row = one party in one election.

The raw ParlGov and World Bank files are not included. See `data/raw/README.md` for how to download them. You only need them to rebuild the table from scratch; every other notebook uses the clean table, which is included.

---

## Method

1. **Building the dataset (notebook 00).** I filtered ParlGov to national elections in Europe and removed placeholder rows such as "others" and "no party affiliation". Then I added each party's results at the two previous elections, whether it was in the outgoing government or held the prime minister, its political family and left–right position, and the economy **of the year before the election**. Using the year before avoids leakage, because an election in June cannot know that year's full figures.
2. **Exploration (notebook 01).** This step set the baseline ("same as last time", 3.59 points) and showed that the previous result is the strongest predictor. It also found two data problems:
   - Parties that did not exist two elections earlier showed a fake "surge", because their old score was recorded as 0.
   - A few hyperinflation values went up to 951%. I capped inflation at 50%.
3. **Feature engineering.** I fixed the fake surge with a `was_new_last_time` flag and added two election-level features: `n_new_parties` and `prev_total_running` (the share of the last vote held by the parties running again).
4. **Validation.** I split the data by time: the models train on 1991–2014 and are tested on 2015–2023, like predicting a real future election. Inside the training years, hyperparameters are tuned with **GridSearchCV + GroupKFold** (5 folds). GroupKFold keeps all parties of one election in the same fold. When two settings score almost the same, I keep the simplest one: the smallest depth within 1% of the best cross-validation score.
5. **Models (notebooks 02–03).** I compared Linear Regression, KNN, a Decision Tree, Bagging, Random Forest, AdaBoost and Gradient Boosting. I measured them with MAE (the main metric), RMSE and R², and checked train vs test error for overfitting. Notebook 02 also tries feature selection by correlation, scaling (standard and min-max) and permutation importance.
   - **Final model:** Gradient Boosting with absolute-error loss, 400 trees, learning rate 0.03 and max depth 4.
6. **France 2027 with the party model (notebook 04).** I built 2027 rows from the 2024 results, compared the predictions with the polls, and tested how sensitive they are to my assumptions.
7. **Bloc model (notebook 05).** I grouped parties into 6 blocs: Far left, Left, Centre, Right, Far right and Other. Alliances that mix two blocs (NUPES 2022, NFP 2024) are split by each bloc's share of the alliance's candidates. I retrained Gradient Boosting on the blocs, merged the result with the polls, and produced the final 2027 estimate with an "8 in 10" range.

---

## Repository structure

```
election-vote-share-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── party_elections.csv                 # clean table (output of notebook 00)
│   ├── sondages-presidentielle-2027.csv    # Sondax 2027 polls
│   └── raw/README.md                       # how to download ParlGov + World Bank files
├── notebooks/
│   ├── 00_building_the_dataset.ipynb       # raw files → clean table, step by step
│   ├── 01_data_exploration.ipynb           # distributions, correlations, baseline
│   ├── 02_modelling.ipynb                  # baselines, linear, RF, GB, feature selection, errors
│   ├── 03_trees_and_ensembles.ipynb        # tree depth, Bagging, RF, AdaBoost, GB, comparison
│   ├── 04_france_2027_projection.ipynb     # party model on France 2027 vs polls
│   ├── 05_blocs_and_polls.ipynb            # bloc model + polls → final 2027 estimate
│   └── election_vote_share_full_project.ipynb  # the whole project in one notebook
├── src/
│   ├── build_dataset.py    # raw files → data/party_elections.csv
│   ├── features.py         # feature fixes, time split, baseline
│   ├── modelling.py        # pipeline, cross-validated tuning, metrics, trains the final model
│   ├── blocs.py            # parties → blocs, alliance splits, bloc table
│   └── polls.py            # polls averaged by bloc
├── models/                 # saved models (.joblib)
└── slides/
    └── Election_Vote_Share_presentation.pptx
```

## How to run

```bash
pip install -r requirements.txt

# optional: rebuild the clean table (needs the raw files in data/raw/)
python -m src.build_dataset

# train and save the final party model (test MAE 3.15)
python -m src.modelling

# or open the notebooks, in order (run them from the notebooks/ folder)
jupyter notebook notebooks/
```

Run notebook 02 before notebooks 04 and 05, because they load the model it saves.

---

## Limitations

- **Structural information only.** The model sees past results, who is in power and the economy. It does not see campaigns, candidates, scandals or alliances being made, and those drive the biggest swings.
- **Recent years are harder.** 2015–2023 saw big political change in Europe: new centre parties, a rising far right, and the traditional parties collapsing. For the big political forces, no model clearly beats "same as last time" in those years (bloc level: 5.12 / 5.20 / 5.32 points).
- **Parliament vs presidential elections.** The models learn from parliament elections, but the 2027 polls are presidential. The translation between the two relies on a single election (2022).
- **Approximations.** The split of NUPES/NFP between far left and left is based on each bloc's share of the candidates, so the two left blocs are less certain than their total. The 50/50 blend of model and polls is a simple choice that was not tested. The 2026 economic figures are estimates.
- **Ethics.** A forecast like this could influence voters or campaigns if presented as certain. It should always be shown with its error range and its limits.

## Future work

- Add the 2024 French election (and other recent elections) to the dataset properly.
- Predict the **change** in vote share rather than the share itself.
- Add poll data for past elections, so the model can learn how much to trust polls compared with history.
- Model the presidential second round.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter.
