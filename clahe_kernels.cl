
kernel void
histogram(global uchar *data,
          int width,
          int height,
          int ldaNumer,
          int ldaDenom,
          global volatile int *hist,
          int iHist,
          int nBins,
          int min,
          int max)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   if (i >= width || j >= height) return;

   uchar val = data[j*ldaNumer/ldaDenom + i];
   int bin = clamp((int) floor(((float) (val - min)) * nBins / (max - min)),
                   (int) 0,
                   (int) (nBins - 1));
   /* printf("%d %d %f %d\n", min, max, floor(((float) (val - min)) * nBins / (max - min)), bin); */
   atomic_inc(&hist[iHist * nBins + bin]);
}

kernel void
equalize(global uchar *in,
         int width,
         int height,
         int ldaNumer,
         int ldaDenom,
         int min,
         int max,
         global int *cdf,
         int iCdf,
         int nBins,
         global uchar *out,
         float normalizationFactor)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   if (i >= width || j >= height)
   {
      out[j*ldaNumer/ldaDenom + i] = in[j*ldaNumer/ldaDenom + i];
      return;
   }

   uchar val = in[j*ldaNumer/ldaDenom + i];
   int bin = clamp((int) floor(((float) (val - min)) * nBins / (max - min)),
                   (int) 0,
                   (int) (nBins - 1));
   /* printf("%d %d %f %d\n", min, max, floor(((float) (val - min)) * nBins / (max - min)), bin); */
   uchar newVal = cdf[iCdf * nBins + bin] * normalizationFactor;
   out[j*ldaNumer/ldaDenom + i] = newVal;

   /* int i = get_global_id(0); */

   /* if (i > size) */
   /*    return; */

   /* uchar val = in[i]; */
   /* int bin = clamp((int) floor(((float) (val - min)) * nBins / (max - min)), */
   /*                 (int) 0, */
   /*                 (int) (nBins - 1)); */
   /* /\* int bin = in[i];//( - min) * nBins / (max - min); *\/ */
   /* uchar newVal = cdf[bin] * maxOut / size; */
   /* out[i] = newVal; */
}

kernel void
clipHistogram(global int *hist,
              int iHist,
              int nBins,
              int maxCounts,
              global int *excess)
{
   int i = get_global_id(0);

   if (i >= nBins)
      return;

   int iShifted = i + iHist * nBins;
   int val = hist[iShifted];
   int binExcess = val - maxCounts;
   if (binExcess > 0)
      atomic_add(&excess[iHist], binExcess);

   hist[iShifted] = min(val, maxCounts);
}

kernel void
addExcess(global int *hist,
          int iHist,
          int nBins,
          int maxCounts,
          const global int *excess,
          int minBin,
          int maxBin)
{
   int i = get_global_id(0);

   if (i >= nBins || i < minBin || i > maxBin)
      return;

   int iShifted = i + iHist * nBins;
   int ex = excess[iHist];
   hist[iShifted] += ex / (maxBin - minBin);

   // Ensure that the counts stay the same
   if (i < ex % (maxBin - minBin))
      hist[iShifted]++;
}

kernel void
localHistogram(global uchar *data,
               int width,
               int height,
               int nWidthTiles,
               int tileWidth,
               int tileHeight,
               global volatile int *hists,
               int nBins,
               int min,
               int max)
{
   int i = get_global_id(1);
   int j = get_global_id(0);

   if (i >= width || j >= height) return;

   int iTile = i / tileWidth;
   int jTile = j / tileHeight;

   global volatile int *hist = &hists[nBins * (jTile * nWidthTiles + iTile)];

   uchar val = data[i + j * width];
   int bin = clamp((int) floor(((float) (val - min)) * nBins / (max - min)),
                   (int) 0,
                   (int) (nBins - 1));
   atomic_inc(&hist[bin]);
}

kernel void
localEqualize(global const uchar *data,
              int width,
              int height,
              int nWidthTiles,
              int nHeightTiles,
              uchar xWrap,
              uchar yWrap,
              int valMin,
              int valMax,
              global const int *loCdfs,
              global const int *hiCdfs,
              int nBins,
              global uchar *out,
              float normalizationFactor)
{
   int i = get_global_id(0);
   int j = get_global_id(1);

   int2 x = (int2) (i, j);

   if (i >= width || j >= height) return;

   uchar val = data[j * width + i];
   float fbin = (float) (val - valMin) * nBins / (valMax - valMin);
   int bin = clamp((int) floor(fbin),
                   (int) 0,
                   (int) (nBins - 1));
   float rem = clamp(fbin - bin, 0.0f, 1.0f);

   int tileWidth = width / nWidthTiles;
   int tileHeight = height / nHeightTiles;

   int2 tileSize = (int2) (tileWidth, tileHeight);

   /* Compute shifted position.  This position is used to find the tiles that
    * the point cares about. */
   int2 sx = x - tileSize/2;

   /* Compute the shifted position in units of tile size */
   float2 ssx = convert_float2(sx) / convert_float2(tileSize);


   int2 periodic = (int2) ((int) xWrap, (int) yWrap);

   /* NOTE: vector comparisons result in -1 for true components and 0 for false,
    * which requires extra minus signs. */
   int2 tileLo = convert_int2(floor(ssx))
      - ((ssx < (float2) (0.0f, 0.0f))
         * ((int2) (1, 1)
            + periodic * (int2) (nWidthTiles - 1, nHeightTiles - 1)));
   int2 tileHi = convert_int2(ceil(ssx))
      + ((ssx > (float2) ((float) nWidthTiles - 1, (float) nHeightTiles - 1))
         * ((int2) (1, 1)
            + periodic * (int2) (nWidthTiles - 1, nHeightTiles - 1)));

   float2 t = clamp(ssx - floor(ssx), (float2) (0.0f, 0.0f), (float2) (1.0f, 1.0f));

   global const int *loUpperLeft = &loCdfs[nBins * (tileLo.y * nWidthTiles + tileLo.x)];
   global const int *loUpperRight = &loCdfs[nBins * (tileLo.y * nWidthTiles + tileHi.x)];
   global const int *loLowerLeft = &loCdfs[nBins * (tileHi.y * nWidthTiles + tileLo.x)];
   global const int *loLowerRight = &loCdfs[nBins * (tileHi.y * nWidthTiles + tileHi.x)];

   global const int *hiUpperLeft = &hiCdfs[nBins * (tileLo.y * nWidthTiles + tileLo.x)];
   global const int *hiUpperRight = &hiCdfs[nBins * (tileLo.y * nWidthTiles + tileHi.x)];
   global const int *hiLowerLeft = &hiCdfs[nBins * (tileHi.y * nWidthTiles + tileLo.x)];
   global const int *hiLowerRight = &hiCdfs[nBins * (tileHi.y * nWidthTiles + tileHi.x)];

   float fval = (((1 - rem) * loUpperLeft[bin] + rem * hiUpperLeft[bin]) * (1 - t.x) * (1 - t.y)
                 + ((1 - rem) * loUpperRight[bin] + rem * hiUpperRight[bin]) * (t.x) * (1 - t.y)
                 + ((1 - rem) * loLowerLeft[bin] + rem * hiLowerLeft[bin]) * (1 - t.x) * (t.y)
                 + ((1 - rem) * loLowerRight[bin] + rem * hiLowerRight[bin]) * (t.x) * (t.y));

   uchar newVal = fval * normalizationFactor;
   out[j * width + i] = newVal;
}

/* This routine computes a variance map of a video.  This is basically a measure
 * of the variance in some video per pixel (or at least in some local area).
 * The idea is that this will give a measure of how likely new pixel data
 * matches the pattern of previous data. */
kernel
void
imageStats(global const float *values,
           int width, int height,
           float fmaAlpha,
           global float *mean,
           global float *squareMean,
           global float *variance)
{
   int2 pixel = (int2) (width, height);

   if (pixel.x >= width || pixel.y >= height)
      return;

   int idx = pixel.x + pixel.y * width;

   float meanVal = mean[idx];
   meanVal = meanVal * fmaAlpha + (1.0f - fmaAlpha) * values[idx];
   mean[idx] = meanVal;

   float square = values[idx];
   square *= square;
   float squareMeanVal = squareMean[idx];
   squareMeanVal = squareMeanVal * fmaAlpha + (1.0f - fmaAlpha) * square;
   squareMean[idx] = squareMeanVal;

   variance[idx] = squareMeanVal - meanVal * meanVal;
}
