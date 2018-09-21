
#ifndef TYPE_A
#define TYPE_A uchar
#endif

#ifndef TYPE_B
#define TYPE_B float
#endif

kernel
void
convertAToB(global TYPE_A *in,
            global TYPE_B *out,
            TYPE_A preScale,
            char preOffsetSign,
            TYPE_A preOffset,
            TYPE_B postScale,
            char postOffsetSign,
            TYPE_B postOffset,
            int size)
{
   int i = get_global_id(0);
   if (i >= size) return;

   TYPE_A val_a = in[i] * preScale + preOffsetSign * preOffset;
   TYPE_B val_b = postScale * ((TYPE_B) val_a) + postOffsetSign * postOffset;
   out[i] = val_b;
}


kernel
void
convertBToA(global TYPE_B *in,
            global TYPE_A *out,
            TYPE_B preScale,
            char preOffsetSign,
            TYPE_B preOffset,
            TYPE_A postScale,
            char postOffsetSign,
            TYPE_A postOffset,
            int size)
{
   int i = get_global_id(0);
   if (i >= size) return;

   TYPE_B val_a = in[i] * preScale + preOffsetSign * preOffset;
   TYPE_A val_b = postScale * ((TYPE_A) val_a) + postOffsetSign * postOffset;
   out[i] = val_b;
}
