# Input data

## Station prices and load

The included annual DE-LU series contains hourly load and day-ahead prices from Open Power System Data, time-series release 2020-10-06 (DOI 10.25832/time_series/2020-10-06). The experiments use the 361 complete UTC days of 2019. The file `opsd_de_lu_2019.csv` is the annual input and the daily profile CSVs support the original station controls. [OPSD data and source attribution](https://data.open-power-system-data.org/time_series/2020-10-06/)

Historical price tests use the signed day-ahead price in EUR/MWh divided by 1,000. The reference reliability and degradation experiments floor the price at 1 EUR/MWh and set the selling price to 98% of the buying price. The reference station load is scaled to 390–475 kW. Separate scenarios examine lower load and photovoltaic surplus.

## Caltech charging sessions

For the fleet experiments, obtain session records for the Caltech site through the ACN-Data portal. The revised experiments select connections from January 1 through March 31, 2019, in Los Angeles local time. Keep the connection, disconnection and delivered-energy fields, with one row per session. The model reads the CSV from `caltech.csv` in the repository root. [ACN-Data access and API instructions](https://ev.caltech.edu/dataset) · [Dataset paper](https://doi.org/10.1145/3307772.3328313)

| Meaning | CSV field | Accepted compact-export name |
| --- | --- | --- |
| Connection time | `connectionTime` | `Start` |
| Disconnection time | `disconnectTime` | `End` |
| Delivered energy on the AC side, in kWh | `kWhDelivered` | `Energy` |

Use timestamps consistently in UTC with an explicit offset, or use local times in `America/Los_Angeles` with no timezone suffix. The revised fleet model filters for at least 2 kWh of delivered energy and builds a pool of feasible, same-day sessions. It uses the actual connected fraction of each hour and sets the battery-energy increment to 0.95 times delivered AC energy. Battery starting energy, discharge capability, power limits and charging-capacity coefficients are model assumptions. The archived independent-fleet notebook uses its own original session and battery rules.
