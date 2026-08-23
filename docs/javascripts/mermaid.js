document$.subscribe(() => {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "strict" });
    window.mermaid.run({ querySelector: ".mermaid" });
  }
});
