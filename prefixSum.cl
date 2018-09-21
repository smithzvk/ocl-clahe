
/* A simple implementation of an inclusive prefix sum.  Not work efficient.
 * Runs in O(log N) time.  */

#ifndef PREFIX_SUM_INFIX_OP
#define PREFIX_SUM_INFIX_OP 1
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
#if PREFIX_SUM_INFIX_OP
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
