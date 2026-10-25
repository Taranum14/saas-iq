/* Draws the Plotly charts. Each chart arrives with colour tokens ("@ink", "@c1", ...) that are
   replaced by the page's CSS variables, so charts follow light/dark mode and the shared palette. */
(function () {
  var cfg = { displayModeBar: false, responsive: true };

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
  }

  function resolve(x) {
    if (typeof x === "string") return x.charAt(0) === "@" ? token(x.slice(1)) || x : x;
    if (Array.isArray(x)) return x.map(resolve);
    if (x && typeof x === "object") {
      var o = {};
      for (var k in x) o[k] = resolve(x[k]);
      return o;
    }
    return x;
  }

  function drawAll() {
    (window.CHARTS || []).forEach(function (c) {
      var el = document.getElementById(c[0]);
      if (!el || !window.Plotly) return;
      var fig = resolve(c[1]);
      window.Plotly.react(el, fig.data, fig.layout, cfg);
    });
  }

  var ready = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  ready.then(drawAll);

})();
