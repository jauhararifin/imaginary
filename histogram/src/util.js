
export async function calculateHistogram(imageSource) {
  const result = {red: [], green: [], blue: [], grayscale: []};
  for (let i = 0; i < 256; i++) {
    result.red.push(0);
    result.green.push(0);
    result.blue.push(0);
    result.grayscale.push(0);
  }

  if (!imageSource) {
    return result;
  }

  const image = new Image();
  image.src = imageSource;
  await new Promise(resolve => image.onload = resolve);

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = image.width;
  canvas.height = image.height;
  context.drawImage(image, 0, 0, image.width, image.height);
  const pixelData = context.getImageData(0, 0, image.width, image.height).data;

  const channelSize = pixelData.length / (image.width * image.height);
  for (let pix = 0; pix < image.width * image.height; pix++) {
    const red = pixelData[pix * channelSize];
    const green = pixelData[pix * channelSize + 1];
    const blue = pixelData[pix * channelSize + 2];
    const grayscale = Math.round((red + green + blue) / 3.0);

    result.red[red]++;
    result.green[green]++;
    result.blue[blue]++;
    result.grayscale[grayscale]++;
  }
  return result;
}