//! f32 math for the CUDA build (frac-rs vendoring): `crate::fm::*f` calls routed to the `f32` methods,
//! which cuda-oxide lowers to libdevice (`__nv_sqrtf`, ...) and the host to the platform libm.
//! libm 0.2.16's x86 `sqrtf` uses an SSE intrinsic the device backend cannot translate.
#![allow(missing_docs)]

#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}
#[inline(always)]
pub fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}
#[inline(always)]
pub fn expf(x: f32) -> f32 {
    x.exp()
}
#[inline(always)]
pub fn sinf(x: f32) -> f32 {
    x.sin()
}
#[inline(always)]
pub fn cosf(x: f32) -> f32 {
    x.cos()
}
#[inline(always)]
pub fn tanf(x: f32) -> f32 {
    x.tan()
}
#[inline(always)]
pub fn acosf(x: f32) -> f32 {
    x.acos()
}
#[inline(always)]
pub fn atanf(x: f32) -> f32 {
    x.atan()
}
#[inline(always)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    y.atan2(x)
}
#[inline(always)]
pub fn floorf(x: f32) -> f32 {
    x.floor()
}
