import { createTheme, type MantineColorsTuple } from "@mantine/core";

const cyan: MantineColorsTuple = [
  "#e0fbfd",
  "#d1f4f8",
  "#a3e7ef",
  "#71dae6",
  "#42d5df",
  "#2cc4cf",
  "#21a9b6",
  "#1c8894",
  "#166873",
  "#124b54",
];

const amber: MantineColorsTuple = [
  "#fdf3df",
  "#fae3bb",
  "#f5cd85",
  "#f2c05c",
  "#efb44b",
  "#e5a234",
  "#c98a24",
  "#a56e1b",
  "#825414",
  "#5e3b0f",
];

const coral: MantineColorsTuple = [
  "#ffe7e5",
  "#ffd1ce",
  "#fba5a1",
  "#f68986",
  "#ef6f66",
  "#dd5b52",
  "#bd483f",
  "#9d3931",
  "#7e2b25",
  "#601f1a",
];

const mint: MantineColorsTuple = [
  "#e2faf0",
  "#cbf2e0",
  "#9ee3c4",
  "#83d8b3",
  "#67d39a",
  "#4fc189",
  "#38a370",
  "#2b855b",
  "#1f6947",
  "#144d34",
];

const dark: MantineColorsTuple = [
  "#dce7ef",
  "#c2d1dc",
  "#8294a5",
  "#566979",
  "#233342",
  "#345066",
  "#111f2e",
  "#0e1a28",
  "#0a1522",
  "#07101b",
];

export const theme = createTheme({
  colors: { cyan, amber, coral, mint, dark },
  primaryColor: "cyan",
  primaryShade: 6,
  defaultRadius: "xs",
  fontFamily: 'Inter, "Segoe UI", sans-serif',
  fontFamilyMonospace: '"IBM Plex Mono", Consolas, monospace',
  headings: {
    fontFamily: '"Barlow Condensed", "Arial Narrow", sans-serif',
    fontWeight: "600",
  },
});
