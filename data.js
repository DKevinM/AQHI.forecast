// data.js
let recentData = [];
let dataByStation = {};
window.dataByStation = dataByStation; 




const unitsLookup = {
  "AQHI": " ", "Ozone": " ppb", "Total Oxides of Nitrogen": " ppb",
  "Hydrogen Sulphide": " ppb", "Total Reduced Sulphur": " ppb", "Sulphur Dioxide": " ppb",
  "Fine Particulate Matter": " µg/m³", "Total Hydrocarbons": " ppm", "Carbon Monoxide": " ppm",
  "Wind Direction": " degrees", "Relative Humidity": " %", "Outdoor Temperature": " °C",
  "Nitric Oxide": " ppb", "Wind Speed": " km/hr", "Non-methane Hydrocarbons": " ppm",
  "Nitrogen Dioxide": " ppb", "Methane": " ppm"
};

const abbrLookup = {
  "AQHI": "AQHI", "Ozone": "O₃", "Total Oxides of Nitrogen": "NOx",
  "Hydrogen Sulphide": "H₂S", "Total Reduced Sulphur": "TRS", "Sulphur Dioxide": "SO₂",
  "Fine Particulate Matter": "PM2.5", "Total Hydrocarbons": "THC", "Carbon Monoxide": "CO",
  "Wind Direction": "wd", "Relative Humidity": "RH", "Outdoor Temperature": "ET",
  "Nitric Oxide": "NO", "Wind Speed": "ws", "Non-methane Hydrocarbons": "NMHC",
  "Nitrogen Dioxide": "NO₂", "Methane": "CH₄"
};

const shortLookup = {
  "AQHI": "AQHI", "Ozone": "O3", "Total Oxides of Nitrogen": "NOX",
  "Hydrogen Sulphide": "H2S", "Total Reduced Sulphur": "TRS", "Sulphur Dioxide": "SO2",
  "Fine Particulate Matter": "PM2.5", "Total Hydrocarbons": "THC", "Carbon Monoxide": "CO",
  "Wind Direction": "wd", "Relative Humidity": "RH", "Outdoor Temperature": "ET",
  "Nitric Oxide": "NO", "Wind Speed": "ws", "Non-methane Hydrocarbons": "NMHC",
  "Nitrogen Dioxide": "NO2", "Methane": "CH4"
};





fetch('https://raw.githubusercontent.com/DKevinM/AB_datapull/main/data/last6h.csv')
  .then(res => res.text())
  .then(text => {
    const rows = text.trim().split('\n');
    const headers = rows.shift().split(',');

    const raw = {};
    rows.forEach(line => {
      const cols = line.split(',');
      const e = Object.fromEntries(headers.map((h,i)=>[h,cols[i]]));
      if (!e.Latitude||!e.Longitude||isNaN(e.Latitude)||isNaN(e.Longitude)) return;

      e.ParameterName = e.ParameterName||"AQHI";
      e.Units = unitsLookup[e.ParameterName]||"";
      e.Abbreviation = abbrLookup[e.ParameterName]||"";
      e.Shortform = shortLookup[e.ParameterName]||"";

      let v = parseFloat(e.Value);
      if (["Ozone","Total Oxides of Nitrogen","Hydrogen Sulphide","Total Reduced Sulphur","Sulphur Dioxide","Nitric Oxide","Nitrogen Dioxide"].includes(e.ParameterName)) {
        v *= 1000;
      }
      if (isNaN(v)) return;
      e.Value = v;

      const utc = new Date(e.ReadingDate);
      e.DisplayDate = utc.toLocaleString("en-CA", {
        timeZone: "America/Edmonton",
        hour12: true
      });
      
      // Keep original ReadingDate as valid ISO string or Date object
      e.ReadingDate = utc.toISOString();


      raw[e.StationName] = raw[e.StationName]||[];
      raw[e.StationName].push(e);
    });

    Object.entries(raw).forEach(([station, arr]) => {
      arr.sort((a, b) => new Date(b.ReadingDate) - new Date(a.ReadingDate));
      const byParam = {};
      arr.forEach(e => {
        const param = e.ParameterName;
        if (!byParam[param] || new Date(e.ReadingDate) > new Date(byParam[param].ReadingDate)) {
          byParam[param] = e;
        }
      });
      dataByStation[station] = Object.values(byParam);
      recentData.push(...dataByStation[station]);
    });
  });


window.fetchAllStationData = function () {
  const stationNames = Object.keys(dataByStation);
  if (stationNames.length === 0) {
    return Promise.resolve([]);
  }

const orderedParams = [
    "AQHI", "Outdoor Temperature", "Relative Humidity", "Wind Speed", "Wind Direction", 
    "Nitrogen Dioxide", "Total Oxides of Nitrogen", "Nitric Oxide", "Ozone",
    "Fine Particulate Matter", "Sulphur Dioxide", "Hydrogen Sulphide", "Total Reduced Sulphur",
    "Carbon Monoxide", "Total Hydrocarbons", "Methane", "Non-methane Hydrocarbons"  
];


const shortformOverride = {
      "Outdoor Temperature": "Temp",
      "Relative Humidity": "Humidity",
      "Wind Speed": "Wind Speed",
      "Wind Direction": "Wind Dir"
};


    // Get latest timestamp from any record
  const latestTimestamp = stationData
    .map(row => new Date(row.ReadingDate))
    .filter(d => !isNaN(d))  // Filter out invalid dates just in case
    .sort((a, b) => b - a)[0]
    .toLocaleString("en-CA", { timeZone: "America/Edmonton", hour12: true });


  const paramLookup = {};
  stationData.forEach(r => {
    paramLookup[r.ParameterName] = r;
  });


const rawTime = stationData[0]?.ReadingDate;

let parsedTime = null;
if (rawTime) {
  parsedTime = new Date(rawTime);  // Don't replace or modify
}

const timestamp = parsedTime && !isNaN(parsedTime.getTime())
  ? parsedTime.toLocaleString("en-CA", {
      timeZone: "America/Edmonton",
      hour12: true
    })
  : "Invalid Date";

console.log("Timestamp:", timestamp);

  
 
  const rows = orderedParams
    .filter(p => paramLookup[p] && p !== "AQHI")
    .map(p => {
      const r = paramLookup[p];
      const label = shortformOverride[p] || r.Shortform || p;
      const value = r.Value;
      const unit = r.Units || "";
      return `${label}: ${value}${unit}`;
    });

  
  const aqhiValue = paramLookup["AQHI"]?.Value || "N/A";

  const html = `
    <div style="font-size:0.9em;">
      <strong>${stationName}</strong><br>
      <small><em>${timestamp}</em></small><br>
      AQHI: ${aqhiValue}<br>
      ${rows.join("<br>")}
    </div>
  `;

  return Promise.resolve(results);
};
