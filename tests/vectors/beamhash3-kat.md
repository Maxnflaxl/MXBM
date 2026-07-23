# BeamHash III — Known-Answer Vectors (the M1 correctness oracle)

Generated from Beam's own reference (`OptimisedSolve`/`IsValidSolution`) and
independently reproduced by a little-endian reimplementation (no `std::bitset`);
both agree bit-for-bit. Beam ships **no** BeamHash III KAT of its own, so these
are MXBM's oracle: Beam's reference implementation is the authority these
vectors were generated from and verified against.

The primitive KATs (§2–§4) need **no** solve — implement against them first.
The full solutions (§5) require enumerating the 2^25 table (~4 min, ~9 GB RAM in
the reference); use the stored vectors, don't regenerate casually.

## 1. Fixed test input

```
input      = 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
             10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f   (32 bytes)
job nonce  = 00 00 00 00 00 00 00 00                          (8 bytes)
extraNonce = 00 00 00 00                                       (4 bytes)
```

## 2. prePow (Blake2b, 4× little-endian u64)

```
prePow[0] = 0xe9c61b07dd959e7f
prePow[1] = 0x249368e9485c8c26
prePow[2] = 0xd28fd445652726b8
prePow[3] = 0xe83213705d078f9f
```

## 3. SipHash-2-4 (non-standard variant — see spec §2)

```
siphash24(prePow, 0)        = 0x10b04caf51290da9
siphash24(prePow, 1)        = 0xe41fc5314952c418
siphash24(prePow, 7)        = 0xb1514093d1c4f4b4
siphash24(prePow, 8)        = 0xc80e82627af2913a
siphash24(prePow, 0xffffff) = 0xe196386a335549bf
# trivial-key sanity (k0,k1,k2,k3 = 0,1,2,3):
siphash24(0,1,2,3, 0)       = 0xdd8748de678c744e
siphash24(0,1,2,3, 1)       = 0x55112546032352d8
```

## 4. Element seed + applyMix(448)

Seed words `w[0..6]` (lo→hi), `w[k] = siphash24(prePow, (index<<3)+k)`:
```
index 0        : 10b04caf51290da9 e41fc5314952c418 7268e57ca49e6ae2 6bc5dfa7f172d033
                 057faed03eb5a8b3 f5a7058be3fd6fc7 0ac7322d4ce874a8
index 1        : c80e82627af2913a aec473643837859c bfe55eeff82d4bae 8d5acb82ebd9513a
                 04fcf8b27ff1e8b8 9f801eb55af1043c f18b55c3c18aad25
index 12345678 : fdc35f2ad0c306f2 310a9228fb4e58ab 0f8a4eb236c695b7 f77d64c565432822
                 a75022d4a38bedbf 5974dd7754794da2 3c57ce805ca3a8b3
```
After `applyMix(448)` on a fresh seed (`indexTree=[index]`), the low word `w[0]`:
```
index 0        -> w[0] = 0x99a961e498026291   collisionBits = 0x026291
index 1        -> w[0] = 0x17a7f7eee7dae097   collisionBits = 0xdae097
index 12345678 -> w[0] = 0xa25ba67c40f60e94   collisionBits = 0xf60e94
```

## 4b. 25-bit index packing round-trip

Input `idx[i] = (i*2654435761) & 0x1FFFFFF` for i=0..31, `pack` then `unpack`
returns the same indices. Packed 100 bytes:
```
00000062f36e88cdbb996833456cdead0eaca28936d3eb29c288cdbb738ee6ab03abd8d4
11c3449ba9bf259aeb29c2af10a0109b7783295ecb399a1f41f0401d58a5709f914d1db17
3f77d98683393c4d5ea6f895eadcecef514a1211989af10a037de5b
```

## 5. Full solutions (104 bytes each) — validated by Beam's `IsValidSolution`

For the §1 input, Beam's `OptimisedSolve` enumerated the full 2^25 table and
emitted exactly these three (each ends `00000000` = extraNonce 0):

```
SOL#1: 182d21b206c9a7ad2bd19fefd51c00359a04ef2e3edac96fd98ceb7bdccc307b
       2b76bd18f69f61d20b5c962d634642056a8bdb5d324262c8872a4d57fa9bbe36
       c5c3244b0be3e96e5ed7ed72e23ceed4dd1d59cf8141d37aeba266e466a5489e
       3708549e00000000

SOL#2: 7b661c7a549b37275a235dad5d6e960e9d06e705eb66750cdfcd7d2cae388973
       8f041bf6ffc60d8c90422cad3a776b1abdf1ddd84f2272dcc85c174313711c49
       2ab306a0b4cf92535492d9519666c22a924f7ed7a43967bd579f12d6dda6947b
       4b2e08cd00000000

SOL#3: eadf012cea5b7c3a80c320705e2342a0dac8d80d891f5f32c1eca2696459117b
       83655b74e559214591cacd2d7f04e6e1a3fa8f1147785fc1ea9c4dd337d61ce5
       8e461d92e81387be1fdfeb1d2a7bb80e88cfaab76d74a99fb8d70a4b5661fd48
       43cb22b400000000
```

These are Equihash-valid regardless of difficulty. M1 exit criterion: MXBM's CPU
solver reproduces these (order-independent) for the §1 input, and MXBM's verifier
accepts all three and rejects any single-bit mutation of them.

> Provenance note: the generator + independent verifier were built this session
> under the scratchpad (`scratchpad/kat/`, ephemeral). The **vectors above are
> the durable artifact**; the tooling is regenerable from the spec and should be
> promoted into `tools/` as the first concrete M1 task (with review).
