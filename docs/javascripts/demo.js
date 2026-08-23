(async () => {
  const app = document.getElementById("demo-app");
  if (!app) return;

  const response = await fetch(app.dataset.url);
  if (!response.ok) {
    app.textContent = "The stored demonstration data could not be loaded.";
    return;
  }
  const data = await response.json();
  const matchSelect = app.querySelector("#demo-match");
  const minuteInput = app.querySelector("#demo-minute");
  const minuteLabel = app.querySelector("#demo-minute-label");
  const scoreline = app.querySelector("#demo-scoreline");
  const margin = app.querySelector("#demo-margin");
  const chart = app.querySelector("#demo-chart");

  for (const match of data.matches) {
    const option = document.createElement("option");
    option.value = String(match.match_id);
    option.textContent = `${match.date} · ${match.home_team} vs ${match.away_team}`;
    matchSelect.appendChild(option);
  }

  const setProbability = (name, value) => {
    app.querySelector(`[data-probability="${name}"]`).textContent = `${(value * 100).toFixed(1)}%`;
    app.querySelector(`[data-bar="${name}"]`).style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
  };

  const renderChart = (points) => {
    const width = 760;
    const height = 270;
    const left = 38;
    const top = 18;
    const plotWidth = 700;
    const plotHeight = 210;
    const colors = { home_win: "#00796b", draw: "#ffb300", away_win: "#5c6bc0" };
    const polyline = (key) => points.map((point) => {
      const x = left + (point.minute / 90) * plotWidth;
      const y = top + (1 - point.probabilities[key]) * plotHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const grid = [0, 0.25, 0.5, 0.75, 1].map((value) => {
      const y = top + (1 - value) * plotHeight;
      return `<line x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}" stroke="currentColor" opacity="0.12"/><text x="2" y="${y + 4}" font-size="11" fill="currentColor">${Math.round(value * 100)}%</text>`;
    }).join("");
    const lines = Object.entries(colors).map(([key, color]) =>
      `<polyline points="${polyline(key)}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round"/>`
    ).join("");
    chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Outcome probability timeline">${grid}${lines}<line x1="${left}" y1="${top + plotHeight}" x2="${left + plotWidth}" y2="${top + plotHeight}" stroke="currentColor" opacity="0.35"/><text x="${left}" y="255" font-size="11" fill="currentColor">0 min</text><text x="${left + plotWidth - 35}" y="255" font-size="11" fill="currentColor">90 min</text></svg>`;
  };

  const render = () => {
    const match = data.matches.find((item) => item.match_id === Number(matchSelect.value));
    const point = match.timeline.find((item) => item.minute === Number(minuteInput.value));
    minuteLabel.textContent = `${point.minute} min`;
    scoreline.textContent = `${match.home_team} ${point.score.home}–${point.score.away} ${match.away_team}`;
    margin.textContent = `${point.expected_final_margin >= 0 ? "+" : ""}${point.expected_final_margin.toFixed(2)} goals`;
    setProbability("home_win", point.probabilities.home_win);
    setProbability("draw", point.probabilities.draw);
    setProbability("away_win", point.probabilities.away_win);
    renderChart(match.timeline);
  };

  matchSelect.addEventListener("change", render);
  minuteInput.addEventListener("input", render);
  matchSelect.value = String(data.matches[0].match_id);
  render();
})();
