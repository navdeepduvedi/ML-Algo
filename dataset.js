export async function loadIrisDataset() {
  const response = await fetch(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
  );
  const csvText = await response.text();

  // Parse CSV data into an array
  const rows = csvText.split("\n").slice(1); // Skip header row
  let data = [];

  for (let row of rows) {
    const values = row.split(",");
    if (values.length < 5) continue; // Skip empty rows

    // Extract numerical values (ignore the species column)
    let features = values.slice(0, 4).map(Number);
    data.push(features);
  }

  return data;
}
