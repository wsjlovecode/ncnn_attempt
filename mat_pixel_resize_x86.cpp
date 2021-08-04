void resize_bilinear(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h){
    return resize_bilinear(src, srcw, srch, dst, w, h, w);
}

void resize_bilinear(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_BITS = 1 << INTER_RESIZE_COEF_BITS;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;
    int* yofs = buf;

    short* ialpha = (short*)(buf + w + h);
    short* ibeta = (short*)(buf + w + h + w);

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for(int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if(sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if(sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_BITS;
        float a1 = fx * INTER_RESIZE_COEF_BITS;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for(int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if(sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if(sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_BITS;
        float b1 = fy * INTER_RESIZE_COEF_BITS;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for(int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if(sy == prev_sy1)
        {
            //
        }
        else if(sy == prev_sy1 + 1)
        {
            // hresize one row
            short* row0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for(int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialpha[0];
                short a1 = ialpha[1];

                const unsigned char* S1p = S1 + sx;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialpha += 2;
            }
        }
        else
        {
            //hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for(int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy;

        //vresize
        int b0 = (int)ibeta[0];         // 使其成为32bit数
        int b1 = (int)ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = w >> 2;
        int remain = w - (nn << 2);

        __m128i _b0 = _mm_set1_epi32(b0);    // 需要修改，用set指令
        __m128i _b1 = _mm_set1_epi32(b1);    // 需要修改，用set指令
        __m128i _v2 = _mm_set1_epi32(2);
        for(; nn > 0; nn--)
        {
            //__mm128i _rows0p_sr4 = _mm_loadl_epi64(rows0p);
            //__mm128i _rows1p_sr4 = _mm_loadl_epi64(rows1p);
            __m128i rows0p_0_sr4 = _mm_set_epi32(*rows0p << 16, 0, *(rows0p+2) << 16, 0);
            __m128i rows0p_1_sr4 = _mm_set_epi32(*(rows0p+1) << 16, 0, *(rows0p+3) << 16, 0);
            __m128i rows1p_0_sr4 = _mm_set_epi32(*rows1p << 16, 0, *(rows1p+2) << 16, 0);
            __m128i rows1p_1_sr4 = _mm_set_epi32(*(rows1p+1) << 16, 0, *(rows1p+3) << 16, 0);
            // __mm128i _rows0p_1_sr4 = _mm_load_epi64(rows0p + 4);
            // __mm128i _rows1p_1_sr4 = _mm_load_epi64(rows1p + 4);

            __m128i _rows0p_0_sr4_mb0 = _mm_mul_epu32(rows0p_0_sr4, _b0);
            __m128i _rows0p_1_sr4_mb0 = _mm_mul_epu32(rows0p_1_sr4, _b0);
            __m128i _rows1p_0_sr4_mb1 = _mm_mul_epu32(rows1p_0_sr4, _b1);
            __m128i _rows1p_1_sr4_mb1 = _mm_mul_epu32(rows1p_1_sr4, _b1);
            // __mm128 _rows0p_1_sr4_mb0 = _mm_mullo_epi16(_rows0p_1_sr4, _b0);
            // __mm128 _rows1p_1_sr4_mb1 = _mm_mullo_epi16(_rows1p_1_sr4, _b1);

            // right shift & pack
            // __m128i _acc = _v2;
            // _acc = _mm_add_epi32(_mm_srli_epi32(_rows0p_0_sr4_mb0, 48), _acc);
            // _acc = _mm_add_epi32(_mm_srli_epi32(_rows1p_sr4_mb1, 48), _acc);
            __m128i rows0p_0_unpack = _mm_unpacklo_epi32(_mm_srli_epi64(_rows0p_0_sr4_mb0, 32), _mm_srli_epi64(_rows0p_1_sr4_mb0, 32));
            __m128i rows1p_0_unpack = _mm_unpacklo_epi32(_mm_srli_epi64(_rows1p_0_sr4_mb1, 32), _mm_srli_epi64(_rows1p_1_sr4_mb1, 32));
            __m128i rows0p_1_unpack = _mm_unpackhi_epi32(_mm_srli_epi64(_rows0p_0_sr4_mb0, 32), _mm_srli_epi64(_rows0p_1_sr4_mb0, 32));
            __m128i rows1p_1_unpack = _mm_unpackhi_epi32(_mm_srli_epi64(_rows1p_0_sr4_mb1, 32), _mm_srli_epi64(_rows1p_1_sr4_mb1, 32));
            __m128i rows0p_pack = _mm_unpacklo_epi64(rows0p_0_unpack, rows0p_1_unpack);
            __m128i rows1p_pack = _mm_unpacklo_epi64(rows1p_0_unpack, rows1p_1_unpack);
            __m128i _acc = _v2;
            _acc = _mm_add_epi32(rows0p_pack, _acc);
            _acc = _mm_add_epi32(rows1p_pack, _acc);
            
            // 右移指令, 并且将int32转化成int8
            __m128i _acc16 = _mm_srli_epi32(_acc, 2);
            
            *(Dp) = (unsigned char)_mm_extract_epi8(_acc16, 0);
            *(Dp+1) = (unsigned char)_mm_extract_epi8(_acc16, 1);
            *(Dp+2) = (unsigned char)_mm_extract_epi8(_acc16, 2);
            *(Dp+3) = (unsigned char)_mm_extract_epi8(_acc16, 3);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        for(; remain; --remain)
        {
            *Dp++ = (unsigned char)(((short)(b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2;
        }

        ibeta += 2;
    }

    delete[] buf;
}