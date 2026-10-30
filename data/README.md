# Data files

| file | contents |
| :--- | :--- |
| `powheg_total_gauss_params_13TeV.csv.gz` | 4,999 POWHEG spectra (the source data) |
| `powheg_total_gauss_params_13TeV_cleaned.csv.gz` | the same, without the 10 anomalous spectra listed below |
| `heft_wide_gauss_testdata_13TeV.csv` | the holdout spectra, written by `code/heft_prepare_data.ipynb` |
| `heft_wide_gauss_traindata_13TeV.csv.gz` | the training rows, written by `code/heft_prepare_data.ipynb` |

`heft_prepare_data.ipynb` reads the full file by default; set `USE_CLEANED = True` there (and in `heft_checks.ipynb`) to use the cleaned copy.

## The POWHEG files

Each row is one spectrum at 13 TeV (POWHEG, NLO):

* columns 1–5: the HEFT Wilson coefficients κλ, c_t, c_tt, c_ggh, c_gghh (the header calls the fourth one `CGHH`);
* columns `0` … `99`: the cross section in pb in 15 GeV bins of m_hh; bin k covers [15k, 15(k+1)) GeV;
* column `ID` (last): the spectrum's identifier, 1, 2, …, 4999. It follows the spectrum through every file of the pipeline (holdout and training files).

The `ID` column was added on Oct 29, 2026: the IDs are the row numbers of the original file, starting at 1. Nothing else in the file was changed; every row's text is identical to the original apart from the appended `,ID`.

## Each spectrum is a histogram of 40,000 equal-weight events

Every bin of every spectrum is an exact whole number of a per-spectrum unit, the weight of one event (largest departure from a whole number: 0.0009 events). Dividing a spectrum's total by its unit counts its events: none has more than 40,000, the median is 39,524 in 0–1500 GeV, and the shortfall from 40,000 is the events above 1500 GeV, which the histogram does not record (rank correlation of the shortfall with the content of the top bin: +0.97). The bin-to-bin fluctuations agree with counting noise from about 40,000 events. See `docs/report/dihiggs_report.pdf`, section 5.

## The anomalous spectra (removed in the cleaned copy)

Ten spectra contain far fewer than 40,000 events, even allowing for the events above 1500 GeV; they look like incomplete runs, and their statistical noise is much larger than that of the others.

Criterion: (events counted in 0–1500 GeV) + (events above 1500 GeV, estimated by extrapolating the last 10 bins with an exponential) < 37,500. The values of this sum have a gap between 36,306 and 37,918; the threshold sits in it. Above the gap the values form a continuous tail, which reflects the imprecision of the overflow estimate.

| ID | events in 0–1500 GeV | estimated above 1500 GeV |
| ---: | ---: | ---: |
| 2842 | 1,348 | ≈ 0 |
| 4960 | 11,876 | ≈ 20 |
| 985 | 12,078 | ≈ 60 |
| 4851 | 13,808 | ≈ 1,200 |
| 4318 | 19,136 | ≈ 60 |
| 4322 | 23,937 | ≈ 160 |
| 1966 | 24,865 | ≈ 300 |
| 3572 | 25,741 | ≈ 3,300 |
| 3208 | 33,814 | ≈ 1,100 |
| 806 | 34,269 | ≈ 2,000 |

IDs 806 and 3208 are the least clear-cut: to reach 40,000, their overflow would have to be about 2.8 and 5.5 times the extrapolated one. The next spectra above the gap, IDs 4794 and 1029 (about 37,900), are kept.
