//! Comparator module for Telegraph deterministic intents.
//!
//! Implements the network's published reference rule (docs: "Comparator modules"):
//!   code 0 Exact     10000 if miner == truth else 0 (tolerance ignored)
//!   code 1 AbsTol    err = |miner - truth|
//!   code 2 RelTol    err = |miner - truth| * 10000 / max(|truth|, 1), basis points, truncated toward zero
//!   code 3 PassFail  same as Exact
//! AbsTol and RelTol: 10000 at err 0, then 10000 * (2*tol - err) / (2*tol) down to 0 at err >= 2*tol; tol <= 0 means
//! only an exact match scores. The code is read as a 32-bit integer and an unknown code is Exact. Multiplications
//! saturate; the RelTol product is computed in 128 bits. No imports, no allocation beyond the node's spec buffer.
#![no_std]

// Per-family build stamp (a wasm custom section; no effect on behaviour). Each family registers a different binary.
include!("tag.rs");

const ONE: i64 = 10_000;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// A tiny bump region for the node's 16-byte spec buffer. The node frees what it allocated, so reuse is safe.
static mut ARENA: [u8; 4096] = [0; 4096];
static mut TOP: usize = 0;

#[no_mangle]
pub extern "C" fn imw_alloc(size: i32) -> i32 {
    unsafe {
        let n = ((size.max(0) as usize) + 15) & !15;
        if TOP + n > 4096 {
            TOP = 0; // the node frees every buffer after a call; start over rather than fail
        }
        if n > 4096 {
            return 0;
        }
        let p = core::ptr::addr_of_mut!(ARENA) as *mut u8 as usize + TOP;
        TOP += n;
        p as i32
    }
}

#[no_mangle]
pub extern "C" fn imw_free(_ptr: i32, _size: i32) {
    unsafe { TOP = 0 }
}

#[inline]
fn abs_wrap(v: i64) -> i64 {
    // i64::abs in a release build: i64::MIN has no positive counterpart and comes back unchanged.
    v.wrapping_abs()
}

#[inline]
fn linear_decay(err: i64, tol: i64) -> i64 {
    if tol <= 0 {
        return if err == 0 { ONE } else { 0 };
    }
    let window = tol.saturating_mul(2);
    if err >= window {
        return 0;
    }
    ONE.saturating_mul(window - err) / window
}

fn score(code: i64, tol: i64, truth: i64, miner: i64) -> i64 {
    // The difference is computed in wrapping arithmetic like the reference (i64 subtraction in release).
    let diff = abs_wrap(miner.wrapping_sub(truth));
    match code as i32 {
        1 => linear_decay(diff, tol),
        2 => {
            let denom = abs_wrap(truth).max(1);
            let q = (diff as i128) * (ONE as i128) / (denom as i128); // i128 division truncates toward zero
            // The reference takes the quotient only when it fits an int64 and otherwise uses i64::MAX, on BOTH sides:
            // a wrapped (negative) difference can give a quotient far below i64::MIN, which must not be truncated.
            let err = if q > i64::MAX as i128 || q < i64::MIN as i128 { i64::MAX } else { q as i64 };
            linear_decay(err, tol)
        }
        // 0 Exact, 3 PassFail and any unknown code
        _ => {
            if miner == truth {
                ONE
            } else {
                0
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn imw_score_deterministic(truth: i64, miner: i64, spec_ptr: i32) -> i64 {
    let p = spec_ptr as usize as *const u8;
    let (code, tol) = unsafe {
        let mut a = [0u8; 8];
        let mut b = [0u8; 8];
        core::ptr::copy_nonoverlapping(p, a.as_mut_ptr(), 8);
        core::ptr::copy_nonoverlapping(p.add(8), b.as_mut_ptr(), 8);
        (i64::from_le_bytes(a), i64::from_le_bytes(b))
    };
    score(code, tol, truth, miner)
}
