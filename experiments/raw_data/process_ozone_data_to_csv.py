import numpy



def csv_format(lat, lat_tcos):
    s = ""
    lon = -179.375
    for tco in lat_tcos:
        s += "%f,%f,0,%f\n" % (lon, lat, tco)
        lon += 1.25
    return s
f = open('L3_ozone_n7t_19881001.txt', 'r')
header = f.readline()
header = f.readline()
header = f.readline()

csv_file = open('tco.csv', 'w')
lat = -89.5
lat_tcos = []
for line in f:
    for i in range(1, len(line), 3):
        try:
            lat_tcos.append(int(line[i:i+3]))
        except ValueError:
            break
    if "lat = " in line:
        csvstr = csv_format(lat, lat_tcos)
        csv_file.write(csvstr)
        lat += 1.0
        lat_tcos = []

csv_file.close()
f.close()
