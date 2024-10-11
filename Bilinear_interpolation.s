.data
test0: .word 0x3e70a3d7, 0x3e8b4396, 0x3eaa7efa, 0x3f6a7efa  #  0.235   0.272    0.333    0.916
test1: .word 0x3eb74bc7, 0x3fb62824, 0x40d8b6ae, 0x4101f5c3
test2: .word 0x3de353f8, 0x40558937, 0x413ad9e8, 0x408a4674
interpolation: .word 0x3f000000


result1: .string "result1:"
result2: .string "\nresult2:"
result3: .string "\nresult3:"
.text

j main


fp32_to_bf16:
    mv t0, a0
    # exp
    li t6, 0x7F800000
    and t1, t0, t6
    # mantissa
    li t6, 0x007FFFFF
    and t2, t0, t6
 
    # remove the other bit
    srli t2, t2, 16
    slli t2, t2, 16
    # t0 = sign, t1 = exp, t2 = mantissa
    srli t0, t0, 31
    slli t0, t0, 31
    or t0, t0, t1
    or t0, t0, t2
    mv a0, t0
    ret
 
my_clz:
    # input a0
    li t0, 0
    li t1, 31
    li t2, 1

    clz_loop:
    # if (x & (1U << i)) t1--
    sll t3, t2, t1
    and t3, a0, t3
    bne t3, x0, clz_done

    addi t0, t0, 1
    addi t1, t1, -1
    bge t1, x0, clz_loop

    clz_done:
    mv a0, t0
    ret   


bf16_add:
    # input a4, a5
    addi sp, sp, -48
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw s5, 20(sp)
    sw s6, 24(sp)
    sw s7, 28(sp)
    sw s8, 32(sp)
    sw s9, 36(sp)
    sw s10, 40(sp)
    sw ra, 44(sp)

    add s0, a5, x0
    add s1, a4, x0

    li t0, 0x7fffffff
    and t1, s0, t0
    and t2, s1, t0

    blt t2, t1, noswap
    mv t0, s0
    mv s0, s1
    mv s1, t0

    noswap:
    # sign s2 s3
    srli s2, s0, 31
    srli s3, s1, 31
    # mantissa s4 s5
    li t0, 0x7fffff
    li t1, 0x800000
    and t2, s0, t0
    or s4, t2, t1
    and t2, s1, t0
    or s5, t2, t1
    # exp s6 s7
    li t0, 23
    li t1, 0xff
    srl t2, s0, t0
    and s6, t2, t1
    srl t2, s1, t0
    and s7, t2, t1
    # exp diff s8
    sub s8, s6, s7

    srl s5, s5, s8
    # sub or add
    or t0, s2, s3
    bne t0, x0, sub
    add s4, s4, s5
    j ma_exit

    sub:
    sub s4, s4, s5

    ma_exit:
    mv a0, s4
    call my_clz

    # s9 = my_clz output
    mv s9, a0
    mv s10, x0
    li t0, 8
    blt t0, s9, shift_sub
    # shift_add
    li t0, 8
    sub s10, t0, s9
    srl s4, s4, s10
    add s6, s6, s10
    j shift_exit

    shift_sub:
    li t0, 8
    sub s10, s9, t0
    sll s4, s4, s10
    sub s6, s6, s10

    shift_exit:
    li t0, 0x80000000
    and t0, s0, t0
    li t1, 23
    sll t1, s6, t1
    li t2, 0x7fffff
    and t2, s4, t2
    or t0, t0, t1
    or a0, t0, t2

    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    lw s7, 28(sp)
    lw s8, 32(sp)
    lw s9, 36(sp)
    lw s10, 40(sp)
    lw ra, 44(sp)
    addi sp, sp, 48
    ret
 
 
imul8:
    addi sp, sp, -16
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw ra, 12(sp)
    mv s0, a0
    mv s1, a1

    mv s2, x0
    li t2, 8

loop_imul:
    andi t0, s1, 1

    beq t0, x0, no_add

    slli t1, s0, 8
    add s2, s2, t1

no_add:
    srli s1, s1, 1
    addi t2, t2, -1
    srli s2, s2, 1
    beq t2, x0, done_imul

    j loop_imul

done_imul:
    mv a0, s2
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    ret
    



bf16_mul:
    # input a4, a5
    addi    sp, sp, -48
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)
    sw      s7, 28(sp)
    sw      s8, 32(sp)
    sw      s9, 36(sp)
    sw      s10, 40(sp)
    sw      ra, 44(sp)

    # a5, a4 = source data -> s0, s1
    mv      s0, a5
    mv      s1, a4

    # mantissa s2, s3
    li      t0, 0x7fffff
    li      t1, 0x800000
    and     t2, s0, t0
    or      t2, t2, t1
    srli    s2, t2, 16

    and     t2, s1, t0
    or      t2, t2, t1
    srli    s3, t2, 16

    # exp s4, s5
    li      t0, 23
    li      t1, 0xff
    srl     t2, s0, t0
    and     s4, t2, t1
    srl     t2, s1, t0
    and     s5, t2, t1
    add     s4, s4, s5
    addi    s4, s4, -127

    mv      a0, s2
    mv      a1, s3

    call    imul8

    mv      s6, a0
    srli    t0, s6, 15
    andi    s7, t0, 1

    beq     s7, x0, no_shift
    srli    s6, s6, 1
    addi    s4, s4, 1

no_shift:
    srli    s6, s6, 7
    andi    s6, s6, 0x7f

    li      s8, 0
    slli    s4, s4, 23
    slli    s6, s6, 16
    or      s8, s4, s8
    or      s8, s6, s8
    mv      a0, s8

    lw      s0, 0(sp)
    lw      s1, 4(sp)
    lw      s2, 8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    lw      s7, 28(sp)
    lw      s8, 32(sp)
    lw      s9, 36(sp)
    lw      s10, 40(sp)
    lw      ra, 44(sp)
    addi    sp, sp, 48

    ret

     
     

     
        
 main:
    # Load test data
    la s0, test0
    la s1, interpolation
    lw a0, 0(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s2, a0
#-------------------(0,2)--------------
    lw a0, 4(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s3, a0
#-------------------calculate(0,1)---------
    mv    a4, s2
    mv    a5, s3
    call bf16_add
    mv s8 , a0

    

    lw a0, 8(s0)      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s2, a0
#-------------------(2,2)--------------
    lw a0, 12(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s3, a0
#-------------------calculate(1,2)---------
    mv    a4, s2
    mv    a5, s3
    call bf16_add
    mv s9 , a0
#-----------------0,1--------------------#    
    mv  a0, s8      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16
    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s10, a0


#-----------------0,1--------------------#    
    mv  a0, s9      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16
    mv  a5,a0    
    #input a4,a5
    call     bf16_mul  # Convert to BF16
    mv    s11, a0
    
    mv    a4, s10
    mv    a5, s11
    call bf16_add
    mv s9 , a0
    
    la a0, result1
    li a7, 4
    ecall
    mv a0 , s9
    li a7, 2
    ecall

    
    
    
#----------------------------- Load test data ---------------------------#
    la s0, test1
    la s1, interpolation
    lw a0, 0(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s2, a0
#-------------------(0,2)--------------
    lw a0, 4(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s3, a0
#-------------------calculate(0,1)---------
    mv    a4, s2
    mv    a5, s3
    call bf16_add
    mv s8 , a0

    

    lw a0, 8(s0)      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s2, a0
#-------------------(2,2)--------------
    lw a0, 12(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s3, a0
#-------------------calculate(1,2)---------
    mv    a4, s2
    mv    a5, s3
    call bf16_add
    mv s9 , a0
#-----------------0,1--------------------#    
    mv  a0, s8      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16
    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s10, a0


#-----------------0,1--------------------#    
    mv  a0, s9      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16
    mv  a5,a0    
    #input a4,a5
    call     bf16_mul  # Convert to BF16
    mv    s11, a0
    
    mv    a4, s10
    mv    a5, s11
    call bf16_add
    mv s9 , a0
    
    la a0, result2
    li a7, 4
    ecall
    mv a0 , s9
    li a7, 2
    ecall


#----------------------------- Load test data ---------------------------#
    la s0, test2
    la s1, interpolation
    lw a0, 0(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s2, a0
#-------------------(0,2)--------------
    lw a0, 4(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s3, a0
#-------------------calculate(0,1)---------
    mv    a4, s2
    mv    a5, s3
    call bf16_add
    mv s8 , a0

    

    lw a0, 8(s0)      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s2, a0
#-------------------(2,2)--------------
    lw a0, 12(s0)      
    call fp32_to_bf16  # Convert to BF16

    mv  a4,a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16

    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s3, a0
#-------------------calculate(1,2)---------
    mv    a4, s2
    mv    a5, s3
    call bf16_add
    mv s9 , a0
#-----------------0,1--------------------#    
    mv  a0, s8      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16
    mv  a5,a0    
    #input a4,a5
    jal    ra, bf16_mul  # Convert to BF16
    mv    s10, a0


#-----------------0,1--------------------#    
    mv  a0, s9      
    call fp32_to_bf16  # Convert to BF16
    mv  a4, a0
    lw a0, 0(s1)        
   call  fp32_to_bf16  # Convert to BF16
    mv  a5,a0    
    #input a4,a5
    call     bf16_mul  # Convert to BF16
    mv    s11, a0
    
    mv    a4, s10
    mv    a5, s11
    call bf16_add
    mv s9 , a0
    
    la a0, result3
    li a7, 4
    ecall
    mv a0 , s9
    li a7, 2
    ecall
    

     
     
 
     
     