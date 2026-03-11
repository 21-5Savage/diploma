# Bankruptcy Labeling (SEC-based)

This pipeline labels ticker rows using a formal SEC bankruptcy signal:

- Form `8-K` or `8-K/A`
- Item `1.03` ("Bankruptcy or Receivership")

It avoids weak keyword-only detection.

## Script

- [`src/test/bankrupt_companies.py`](/home/batenkh/diploma/src/test/bankrupt_companies.py)

## Input CSV

Required column:
- `ticker`

Optional column:
- `as_of_date` (YYYY-MM-DD). If present, `label` is computed as bankruptcy in `(as_of_date, as_of_date + horizon_months]`.

## Output columns

- `ticker`
- `cik`
- `bankruptcy_filing_date`
- `bankruptcy_form`
- `bankruptcy_item`
- `accession_number`
- `bankrupt` (1 if any observed SEC bankruptcy signal)
- `label` (forward outcome if `as_of_date` exists, else equals `bankrupt`)
- `evidence_source`
- `evidence_url`

## Run

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python src/test/bankrupt_companies.py \
  --input dataset/tickers.csv \
  --output dataset/labelled_tickers.csv \
  --horizon-months 24 \
  --sleep 0.2
```

Or pass `--user-agent` directly.

## Notes

- SEC requires a valid `User-Agent` identifying the requester.
- For survivorship-bias mitigation, use a historical constituents file with `as_of_date` snapshots, not only current S&P members.
