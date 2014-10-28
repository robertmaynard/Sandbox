	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z17single_allocationv
	.align	4, 0x90
__Z17single_allocationv:                ## @_Z17single_allocationv
	.cfi_startproc
## BB#0:
	push	rbp
Ltmp3:
	.cfi_def_cfa_offset 16
Ltmp4:
	.cfi_offset rbp, -16
	mov	rbp, rsp
Ltmp5:
	.cfi_def_cfa_register rbp
	push	r15
	push	r14
	push	rbx
	sub	rsp, 24
Ltmp6:
	.cfi_offset rbx, -40
Ltmp7:
	.cfi_offset r14, -32
Ltmp8:
	.cfi_offset r15, -24
	xor	ebx, ebx
	lea	rdi, qword ptr [rbp - 40]
	xor	esi, esi
	call	_gettimeofday
	movabs	rdi, 17179869184
	call	__Znam
	mov	r15, rax
	mov	r14, qword ptr [rip + __ZSt4cout@GOTPCREL]
	lea	rsi, qword ptr [rip + L_.str]
	mov	edx, 17
	mov	rdi, r14
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movabs	rsi, 4294967296
	mov	rdi, r14
	call	__ZNSo9_M_insertIxEERSoT_
	mov	r14, rax
	mov	rax, qword ptr [r14]
	mov	rdi, qword ptr [rax - 24]
	add	rdi, r14
	mov	esi, 10
	call	__ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc
	movsx	esi, al
	mov	rdi, r14
	call	__ZNSo3putEc
	mov	rdi, rax
	call	__ZNSo5flushEv
	lea	rdi, qword ptr [rbp - 40]
	xor	esi, esi
	call	_gettimeofday
	mov	dword ptr [r15], 0
	jmp	LBB0_1
	.align	4, 0x90
LBB0_2:                                 ##   in Loop: Header=BB0_1 Depth=1
	mov	eax, dword ptr [r15 + 4*rbx - 4]
	add	eax, ebx
	mov	dword ptr [r15 + 4*rbx], eax
	inc	rbx
LBB0_1:                                 ## =>This Inner Loop Header: Depth=1
	cmp	ebx, 1073741823
	jle	LBB0_2
## BB#3:                                ##   in Loop: Header=BB0_1 Depth=1
	mov	eax, ebx
	shr	eax, 31
	add	eax, ebx
	and	eax, -2
	mov	ecx, ebx
	sub	ecx, eax
	mov	dword ptr [r15 + 4*rbx], ecx
	inc	rbx
	jmp	LBB0_1
	.cfi_endproc

	.section	__TEXT,__literal4,4byte_literals
	.align	2
LCPI1_0:
	.long	1232348160              ## float 1.0E+6
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z20multiple_allocationsv
	.align	4, 0x90
__Z20multiple_allocationsv:             ## @_Z20multiple_allocationsv
	.cfi_startproc
## BB#0:
	push	rbp
Ltmp12:
	.cfi_def_cfa_offset 16
Ltmp13:
	.cfi_offset rbp, -16
	mov	rbp, rsp
Ltmp14:
	.cfi_def_cfa_register rbp
	push	r15
	push	r14
	push	r12
	push	rbx
	sub	rsp, 32
Ltmp15:
	.cfi_offset rbx, -48
Ltmp16:
	.cfi_offset r12, -40
Ltmp17:
	.cfi_offset r14, -32
Ltmp18:
	.cfi_offset r15, -24
	xor	ebx, ebx
	lea	rdi, qword ptr [rbp - 48]
	xor	esi, esi
	call	_gettimeofday
	mov	r15, qword ptr [rbp - 48]
	movsxd	r14, dword ptr [rbp - 40]
	mov	edi, 32768
	call	__Znam
	mov	r12, rax
	.align	4, 0x90
LBB1_1:                                 ## =>This Inner Loop Header: Depth=1
	mov	edi, 4194304
	call	__Znam
	mov	qword ptr [r12 + 8*rbx], rax
	inc	rbx
	cmp	rbx, 4096
	jne	LBB1_1
## BB#2:
	xor	ebx, ebx
	lea	rdi, qword ptr [rbp - 48]
	xor	esi, esi
	call	_gettimeofday
	mov	rax, qword ptr [rbp - 48]
	movsxd	rcx, dword ptr [rbp - 40]
	sub	rax, r15
	cvtsi2ss	xmm0, rax
	sub	rcx, r14
	cvtsi2ss	xmm1, rcx
	divss	xmm1, dword ptr [rip + LCPI1_0]
	addss	xmm1, xmm0
	movss	dword ptr [rbp - 56], xmm1 ## 4-byte Spill
	.align	4, 0x90
LBB1_3:                                 ## =>This Inner Loop Header: Depth=1
	mov	rdi, qword ptr [r12 + 8*rbx]
	mov	esi, 97
	mov	edx, 1048576
	call	_memset
	inc	rbx
	cmp	rbx, 4096
	jne	LBB1_3
## BB#4:                                ## %.preheader
	movss	xmm0, dword ptr [rbp - 56] ## 4-byte Reload
	cvtss2sd	xmm0, xmm0
	movsd	qword ptr [rbp - 56], xmm0 ## 8-byte Spill
	xor	ebx, ebx
	.align	4, 0x90
LBB1_5:                                 ## =>This Inner Loop Header: Depth=1
	mov	rdi, qword ptr [r12 + 8*rbx]
	test	rdi, rdi
	je	LBB1_7
## BB#6:                                ##   in Loop: Header=BB1_5 Depth=1
	call	__ZdaPv
LBB1_7:                                 ##   in Loop: Header=BB1_5 Depth=1
	inc	rbx
	cmp	rbx, 4096
	jne	LBB1_5
## BB#8:
	mov	rdi, r12
	call	__ZdaPv
	movsd	xmm0, qword ptr [rbp - 56] ## 8-byte Reload
	add	rsp, 32
	pop	rbx
	pop	r12
	pop	r14
	pop	r15
	pop	rbp
	ret
	.cfi_endproc

	.section	__TEXT,__literal4,4byte_literals
	.align	2
LCPI2_0:
	.long	1232348160              ## float 1.0E+6
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z29inverted_multiple_allocationsv
	.align	4, 0x90
__Z29inverted_multiple_allocationsv:    ## @_Z29inverted_multiple_allocationsv
	.cfi_startproc
## BB#0:
	push	rbp
Ltmp22:
	.cfi_def_cfa_offset 16
Ltmp23:
	.cfi_offset rbp, -16
	mov	rbp, rsp
Ltmp24:
	.cfi_def_cfa_register rbp
	push	r15
	push	r14
	push	r12
	push	rbx
	sub	rsp, 32
Ltmp25:
	.cfi_offset rbx, -48
Ltmp26:
	.cfi_offset r12, -40
Ltmp27:
	.cfi_offset r14, -32
Ltmp28:
	.cfi_offset r15, -24
	xor	ebx, ebx
	lea	rdi, qword ptr [rbp - 48]
	xor	esi, esi
	call	_gettimeofday
	mov	r15, qword ptr [rbp - 48]
	movsxd	r14, dword ptr [rbp - 40]
	mov	edi, 8388608
	call	__Znam
	mov	r12, rax
	.align	4, 0x90
LBB2_1:                                 ## =>This Inner Loop Header: Depth=1
	mov	edi, 16384
	call	__Znam
	mov	qword ptr [r12 + 8*rbx], rax
	inc	rbx
	cmp	rbx, 1048576
	jne	LBB2_1
## BB#2:
	xor	ebx, ebx
	lea	rdi, qword ptr [rbp - 48]
	xor	esi, esi
	call	_gettimeofday
	mov	rax, qword ptr [rbp - 48]
	movsxd	rcx, dword ptr [rbp - 40]
	sub	rax, r15
	cvtsi2ss	xmm0, rax
	sub	rcx, r14
	cvtsi2ss	xmm1, rcx
	divss	xmm1, dword ptr [rip + LCPI2_0]
	addss	xmm1, xmm0
	movss	dword ptr [rbp - 56], xmm1 ## 4-byte Spill
	.align	4, 0x90
LBB2_3:                                 ## =>This Inner Loop Header: Depth=1
	mov	rdi, qword ptr [r12 + 8*rbx]
	mov	esi, 97
	mov	edx, 4096
	call	_memset
	inc	rbx
	cmp	rbx, 1048576
	jne	LBB2_3
## BB#4:                                ## %.preheader
	movss	xmm0, dword ptr [rbp - 56] ## 4-byte Reload
	cvtss2sd	xmm0, xmm0
	movsd	qword ptr [rbp - 56], xmm0 ## 8-byte Spill
	xor	ebx, ebx
	.align	4, 0x90
LBB2_5:                                 ## =>This Inner Loop Header: Depth=1
	mov	rdi, qword ptr [r12 + 8*rbx]
	test	rdi, rdi
	je	LBB2_7
## BB#6:                                ##   in Loop: Header=BB2_5 Depth=1
	call	__ZdaPv
LBB2_7:                                 ##   in Loop: Header=BB2_5 Depth=1
	inc	rbx
	cmp	rbx, 1048576
	jne	LBB2_5
## BB#8:
	mov	rdi, r12
	call	__ZdaPv
	movsd	xmm0, qword ptr [rbp - 56] ## 8-byte Reload
	add	rsp, 32
	pop	rbx
	pop	r12
	pop	r14
	pop	r15
	pop	rbp
	ret
	.cfi_endproc

	.globl	_main
	.align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	push	rbp
Ltmp31:
	.cfi_def_cfa_offset 16
Ltmp32:
	.cfi_offset rbp, -16
	mov	rbp, rsp
Ltmp33:
	.cfi_def_cfa_register rbp
	call	__Z17single_allocationv
	.cfi_endproc

	.section	__TEXT,__StaticInit,regular,pure_instructions
	.align	4, 0x90
__GLOBAL__I_a:                          ## @_GLOBAL__I_a
	.cfi_startproc
## BB#0:
	push	rbp
Ltmp37:
	.cfi_def_cfa_offset 16
Ltmp38:
	.cfi_offset rbp, -16
	mov	rbp, rsp
Ltmp39:
	.cfi_def_cfa_register rbp
	push	rbx
	push	rax
Ltmp40:
	.cfi_offset rbx, -24
	lea	rbx, qword ptr [rip + __ZStL8__ioinit]
	mov	rdi, rbx
	call	__ZNSt8ios_base4InitC1Ev
	mov	rdi, qword ptr [rip + __ZNSt8ios_base4InitD1Ev@GOTPCREL]
	mov	rdx, qword ptr [rip + ___dso_handle@GOTPCREL]
	mov	rsi, rbx
	add	rsp, 8
	pop	rbx
	pop	rbp
	jmp	___cxa_atexit           ## TAILCALL
	.cfi_endproc

.zerofill __DATA,__bss,__ZStL8__ioinit,1,0 ## @_ZStL8__ioinit
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"FULL_BLOCK_SIZE: "

	.section	__DATA,__mod_init_func,mod_init_funcs
	.align	3
	.quad	__GLOBAL__I_a

.subsections_via_symbols
