import { expect, test } from "@playwright/test";

test.describe("browser demo", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("shows the cluster overview and recommended compute path", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: "Cluster overview", exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("main").getByTestId("compute-path")).toBeVisible();
    await expect(page.getByRole("navigation")).toContainText("Nodes");
    await expect(page.getByRole("navigation")).toContainText("Models");
    await expect(page.getByRole("navigation")).toContainText("Chat");
  });

  test("moves between model and network workflows", async ({ page }) => {
    await page.getByRole("button", { name: "Models", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Model library", exact: true })).toBeVisible();
    await expect(page.getByTestId("model-list")).toBeVisible();

    await page.getByRole("button", { name: "Network", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Link diagnostics", exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: /run network test/i }).click();
    await expect(page.getByTestId("network-test-result")).toBeVisible();
  });

  test("accepts a chat prompt through the accessible composer", async ({ page }) => {
    await page.getByRole("button", { name: "Models", exact: true }).click();
    await page.getByRole("button", { name: /launch meridian 12b instruct/i }).click();
    await expect(page.getByRole("status")).toContainText(/is loading/i);

    await page.getByRole("button", { name: "Chat", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Cluster chat", exact: true })).toBeVisible();

    const composer = page.getByTestId("chat-composer");
    const prompt = composer.getByRole("textbox");
    await prompt.fill("Summarize why the recommended compute path was selected.");

    await expect(prompt).toHaveValue("Summarize why the recommended compute path was selected.");
    await expect(composer.getByRole("button", { name: /send/i })).toBeEnabled();
  });
});
