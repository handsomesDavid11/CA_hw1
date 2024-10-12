#include <stdint.h>
#include <stdio.h>

// Define bf16 structure
typedef struct {
    uint16_t bits;
} bf16_t;

// float to bf16
static inline bf16_t fp32_to_bf16(float s) {
    bf16_t h;
    union {
        float f;
        uint32_t i;
    } u = {.f = s};
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* NaN */
        h.bits = (u.i >> 16) | 64;
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 0x10) & 1))) >> 0x10;
    return h;
}

// bf16 to float
static inline float bf16_to_fp32(bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u = {.i = (uint32_t)h.bits << 16};
    return u.f;
}

// Get a specific bit
static inline int32_t getbit(int32_t value, int n) {
    return (value >> n) & 1;
}

// 16-bit integer multiplication
int32_t imul16(int16_t a, int16_t b) {
    int32_t r = 0, a32 = (int32_t) a, b32 = (int32_t) b;
    for (int i = 0; i < 16; i++) {
        if (getbit(b32, i))
            r += a32 << i;
    }
    return r;
}
bf16_t bf16_sub(bf16_t a, bf16_t b) {
    uint16_t ia = a.bits, ib = b.bits;
    int sa = ia >> 15, sb = ib >> 15;
    int32_t ma = (ia & 0x7F) | 0x80, mb = (ib & 0x7F) | 0x80;
    int16_t ea = (ia >> 7) & 0xFF, eb = (ib >> 7) & 0xFF;

    if (ea > eb) {
        mb >>= (ea - eb);
        eb = ea;
    } else if (eb > ea) {
        ma >>= (eb - ea);
        ea = eb;
    }

    int32_t mrtmp = (sa != sb) ? ma + mb : (ma >= mb ? ma - mb : mb - ma);
    if (mrtmp & 0x100) {
        mrtmp >>= 1;
        ea++;
    }

    while (mrtmp && (mrtmp & 0x80) == 0) {
        mrtmp <<= 1;
        ea--;
    }

    uint16_t mr = mrtmp & 0x7F;
    uint16_t er = ea & 0xFF;
    uint16_t r = (sa << 15) | (er << 7) | mr;

    return (bf16_t){.bits = r};
}



// BF16 multiplication
bf16_t bf16_mul(bf16_t a, bf16_t b) {
    uint16_t ia = a.bits, ib = b.bits;
    int sa = ia >> 15, sb = ib >> 15;
    int16_t ma = (ia & 0x7F) | 0x80, mb = (ib & 0x7F) | 0x80;
    int16_t ea = (ia >> 7) & 0xFF, eb = (ib >> 7) & 0xFF;
    
    int32_t mrtmp = imul16(ma, mb);
    int mshift = (mrtmp >> 15) & 1;
    uint16_t mr = mrtmp >> (mshift + 7);
    int16_t ertmp = ea + eb - 127;
    int16_t er = mshift ? ertmp + 1 : ertmp;
    int sr = sa ^ sb;
    uint16_t r = (sr << 15) | ((er & 0xFF) << 7) | (mr & 0x007F);
    
    return (bf16_t){.bits = r};
}





// BF16 addition
static inline int my_clz(uint16_t x) {
    int count = 0;
    for (int i = 15; i >= 0; --i) {
        if (x & (1U << i))
            break;
        count++;
    }
    return count;
}


bf16_t bf16_add(bf16_t a, bf16_t b) {

    
    uint16_t ia = a.bits;
    uint16_t ib = b.bits;

    /* sign */
    int sa = ia >> 15;
    int sb = ib >> 15;

    /* mantissa */
    int16_t ma = (ia & 0x7F) | 0x80;
    int16_t mb = (ib & 0x7F) | 0x80;

    /* exponent */
    int16_t ea = (ia >> 7) & 0xFF;
    int16_t eb = (ib >> 7) & 0xFF;

    // sub_exp and mb shift
    if (ea > eb) {
        mb >>= (ea - eb);
        eb = ea;
    } else if (eb > ea) {
        ma >>= (eb - ea);
        ea = eb;
    }

    //
    int16_t mrtmp;
    if (sa == sb) {
        // add
        mrtmp = ma + mb;
    } else {
        // sub
        if (ma >= mb) {
            mrtmp = ma - mb;
        } else {
            mrtmp = mb - ma;
            sa = sb;  
        }
    }

    //use my clz
    int clz = my_clz(mrtmp);
    //printf("%d",clz);
    int16_t shift = 0;
    if (clz <= 8) {
        shift = 8 - clz;
        mrtmp >>= shift;
        ea += shift;
    } 
    else {
        shift = clz - 8;
        mrtmp <<= shift;
        ea -= shift;
    }

    uint16_t mr = mrtmp & 0x7F;  
    uint16_t er = ea & 0xFF;     


    uint16_t r = (sa << 15) | (er << 7) | mr;

    bf16_t result = {.bits = r};
    return result;
}

int main() {
    //use linear interpolation to find point[1][1]
    float points[3][3][3] = {
    {{0.235 , 0   , 0.272},
     {0     , 0   , 0    },
     {0.333 , 0   , 0.916}},

    {{0.358 , 0   , 1.4231},
     {0     , 0   , 0     },
     {6.7723, 0   , 8.1225}},

    {{0.111 , 0   , 3.3365},
     {0     , 0   , 0     },
     {11.6782, 0  , 4.3211}}
};

for(int i = 0; i<3 ; i++){

    bf16_t p01 =  bf16_add(bf16_mul(fp32_to_bf16((float)0.5),fp32_to_bf16(points[i][0][0])),bf16_mul(fp32_to_bf16((float)0.5),fp32_to_bf16(points[i][0][2])));
//    printf("result1 (Float): %f\n", bf16_to_fp32(p01)); 
    bf16_t p21 =  bf16_add(bf16_mul(fp32_to_bf16((float)0.5),fp32_to_bf16(points[i][2][0])),bf16_mul(fp32_to_bf16((float)0.5),fp32_to_bf16(points[i][2][2])));
//    printf("result1 (Float): %f\n", bf16_to_fp32(p21)); 
    bf16_t p11 =  bf16_add(bf16_mul(fp32_to_bf16((float)0.5),p01),bf16_mul(fp32_to_bf16((float)0.5),p21));

    printf("result1 (Float): %f\n", bf16_to_fp32(p11)); 
}

    return 0;
}
