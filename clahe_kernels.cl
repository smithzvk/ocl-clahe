
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

/* kernel void */
/* minMaxBins(global int *cdf, */
/*            int size, */
/*            int offset, */
/*            global int *out) */
/* { */
/*    int i = get_global_id(0); */

/*    if (i > size) */
/*       return; */

/*    if (i < offset) */
/*       out[i] = in[i]; */
/*    else */
/*       out[i] = in[i] + in[i - offset]; */
/* } */

/* A simple implementation of an inclusive prefix sum.  Not work efficient.  Runs in O(log N) time.  */

#ifndef INFIX_PREFIX_SUM_OP
#define INFIX_PREFIX_SUM_OP 1
#endif

#ifndef PREFIX_SUM_OP
#define PREFIX_SUM_OP +
#endif

#ifndef PREFIX_SUM_DATA_TYPE
#define PREFIX_SUM_DATA_TYPE int
#endif

#ifndef PREFIX_SUM_OP_ZERO
#define PREFIX_SUM_OP_ZERO 0
#endif

int add(int a, int b)
{
   return a + b;
}

kernel void
prefixSum(global PREFIX_SUM_DATA_TYPE *in,
          int start,
          int size,
          int offset,
          global PREFIX_SUM_DATA_TYPE *out)
{
   int i = get_global_id(0);

   if (i >= size)
      return;

   int iShifted = i + start;

   if (i < offset)
      out[iShifted] = in[iShifted];
   else
   {
      if (iShifted - offset < 0)
         printf("Accessing data before the array\n");
#if INFIX_PREFIX_SUM_OP
      out[iShifted] = in[iShifted] PREFIX_SUM_OP in[iShifted - offset];
#else
      out[iShifted] = PREFIX_SUM_OP(in[iShifted], in[iShifted - offset]);
#endif
   }
}

kernel void
prefixSumIncToExc(global PREFIX_SUM_DATA_TYPE *in,
                  int start,
                  int size,
                  global PREFIX_SUM_DATA_TYPE *out)
{
   int i = get_global_id(0);

   if (i >= size)
      return;

   int iShifted = i + start;

   if (i == 0)
      out[iShifted] = PREFIX_SUM_OP_ZERO;
   else
      out[iShifted] = in[iShifted - 1];
}

/* kernel void */
/* stridedPrefixSum(global int *in, */
/*                  int start, */
/*                  int size, */
/*                  int offset, */
/*                  global int *out) */
/* { */
/*    int i = get_global_id(0); */

/*    if (i > size) */
/*       return; */

/*    int iShifted = i + start; */

/*    if (i < offset) */
/*       out[iShifted] = in[iShifted]; */
/*    else */
/*    { */
/*       if (iShifted - offset < 0) */
/*          printf("Accessing data before the array\n"); */
/* #if INFIX_SCA
N_OP */
/*       out[iShifted] = in[iShifted] PREFIX_SUM_OP in[iShifted - offset]; */
/* #else */
/*       out[iShifted] = PREFIX_SUM_OP(in[iShifted], in[iShifted - offset]); */
/* #endif */
/*    } */
/* } */

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
               int nHeightTiles,
               global volatile int *hists,
               int nBins,
               int min,
               int max)
{
   int i = get_global_id(1);
   int j = get_global_id(0);

   if (i >= width || j >= height) return;

   int iTile = i / nWidthTiles;
   int jTile = j / nHeightTiles;

   global volatile int *hist = &hists[nBins * (jTile * nWidthTiles + iTile)];

   uchar val = data[i];
   int bin = clamp((int) floor(((float) (val - min)) * nBins / (max - min)),
                   (int) 0,
                   (int) (nBins - 1));
   /* printf("%d %d %f %d\n", min, max, floor(((float) (val - min)) * nBins / (max - min)), bin); */
   atomic_inc(&hist[bin]);
}

kernel void
localEqualize(global const uchar *data,
              int width,
              int height,
              int nWidthTiles,
              int nHeightTiles,
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


   int2 tileLo = max(convert_int2(floor(ssx)),
                     (int2) (0, 0));
   int2 tileHi = min(convert_int2(ceil(ssx)),
                     (int2) (nWidthTiles-1, nHeightTiles-1));

   float2 t = clamp(ssx - floor(ssx), (float2) (0.0f, 0.0f), (float2) (1.0f, 1.0f));

   global const int *loUpperLeft = &loCdfs[nBins * (tileLo.y * nWidthTiles + tileLo.x)];
   global const int *loUpperRight = &loCdfs[nBins * (tileLo.y * nWidthTiles + tileHi.x)];
   global const int *loLowerLeft = &loCdfs[nBins * (tileHi.y * nWidthTiles + tileLo.x)];
   global const int *loLowerRight = &loCdfs[nBins * (tileHi.y * nWidthTiles + tileHi.x)];

   global const int *hiUpperLeft = &hiCdfs[nBins * (tileLo.y * nWidthTiles + tileLo.x)];
   global const int *hiUpperRight = &hiCdfs[nBins * (tileLo.y * nWidthTiles + tileHi.x)];
   global const int *hiLowerLeft = &hiCdfs[nBins * (tileHi.y * nWidthTiles + tileLo.x)];
   global const int *hiLowerRight = &hiCdfs[nBins * (tileHi.y * nWidthTiles + tileHi.x)];

   float fval = ((rem * loUpperLeft[bin] + (1 - rem) * hiUpperLeft[bin]) * (1 - t.x) * (1 - t.y)
                 + (rem * loUpperRight[bin] + (1 - rem) * hiUpperRight[bin]) * (t.x) * (1 - t.y)
                 + (rem * loLowerLeft[bin] + (1 - rem) * hiLowerLeft[bin]) * (1 - t.x) * (t.y)
                 + (rem * loLowerRight[bin] + (1 - rem) * hiLowerRight[bin]) * (t.x) * (t.y));

   uchar newVal = fval * normalizationFactor;
   out[j * width + i] = newVal;
}
