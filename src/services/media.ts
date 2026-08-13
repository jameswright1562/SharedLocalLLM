export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error(`Could not read ${file.name} as an image.`));
    });
    reader.addEventListener("error", () =>
      reject(reader.error ?? new Error(`Could not read ${file.name}.`)),
    );
    reader.readAsDataURL(file);
  });
}
