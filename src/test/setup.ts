import "@testing-library/jest-dom/vitest";

Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
  configurable: true,
  value: () => undefined,
});

Object.defineProperty(window.navigator, "clipboard", {
  configurable: true,
  value: { writeText: async () => undefined },
});
