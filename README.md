# Mass-conserving multi-task model for crop yield and nitrogen outcomes

ENGG2112 group project (AI for Good, eutrophication). One neural network for soft wheat, barley, grain maize and rapeseed across EU and UK NUTS2 regions (2000-2018) predicts crop yield and harvested N removal, and calculates soil-surface N surplus from the nitrogen budget, so its outputs always add up. It is tested on later years (2017-2018) and on whole countries held out of training.

## Data
The raw data is not in this repository. Download it into `data/raw/capri/`:

- CAPRI disaggregated dataset, Koeble et al. (2026), CC BY 4.0: https://zenodo.org/records/13992301. Get all `.zip` packages and readme files.
- HaNi-Crop, Hao and Tian: https://zenodo.org/records/21364978. Get `HaNi-Crop-<Crop>.nc` for Wheat, Barley, Maize and Rapeseed, plus Rye, Potatoes, Pulses, Sunflower, Soybean and Rice for the external check. The dataset is under peer review. This project uses it with the authors' permission; anyone reusing it should contact them first.

## Setup
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run order
```
.venv/bin/python evaluate.py               # training, blocked tests, metrics, hotspots (about 2 minutes)
.venv/bin/python figures.py                # figures/report and figures/slides
```

## One prediction
```
.venv/bin/python predict.py --list DE
.venv/bin/python predict.py --region DE11 --crop wheat --year 2018
```
This prints the model's yield, removal and surplus next to the budget values, with the error and the budget residual. See `reports/model_and_code_guide.docx` for how the model and code work.

## Layout
| Path | Contents |
|---|---|
| `models.py` | The network, the budget step, the loss and the training loop |
| `evaluate.py` | Splits, experiments, metrics, hotspot ranking |
| `predict.py` | One prediction for a chosen region, crop and year |
| `figures.py` | All figures |
| `results/` | Every number quoted in the reports, as CSV |
