

import requests
import re
import pandas as pd

with open('stations.txt', 'r') as f:
    stations = [line.strip() for line in f.readlines()]

print stations

lonlat_re = re.compile(r'Latitude</b></td><td>([0-9\.]+)&#176N</td><td><b>Longitude</b></td><td>([0-9\.]+)&#176W')
elev_re = re.compile(r"<b>Elevation</b></td><td>([0-9\.]+)' ft</td></tr>")

for station in stations:

    try:
        metadata_url = "http://cdec.water.ca.gov/cgi-progs/staMeta?station_id=%s" % station
        r = requests.get(metadata_url)
        m = lonlat_re.search(r.text)
        lat = -1.0 * float(m.group(1))
        lon = float(m.group(2))

        m = elev_re.search(r.text)
        elev = float(m.group(1))

        url = "http://cdec.water.ca.gov/cgi-progs/queryCSV?station_id=%s&sensor_num=82&dur_code=D&start_date=2011-11-01&end_date=2012-06-01&data_wish=Download+CSV+Data+Now" % station
        r = requests.get(url)

        with open('r.txt', 'w') as f:
            f.write(r.text)
        f = pd.read_csv('r.txt', skiprows=1, na_values=['m'], keep_default_na=True, parse_dates=True, index_col=0, usecols=[0,2])
        f['lon'] = lon
        f['lat'] = lat
        f['elev'] = elev
        f = f.rename(columns={"'SNOW": 'snow'})
        f = f.reindex_axis(['lon', 'lat', 'elev', 'snow'], axis=1)

        f.index.name='date'
        f.to_csv('%s.csv' % station)
        print "saved %s" % station
    except Exception as e:
        print "error at %s:" % station, e
        continue
