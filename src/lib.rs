use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
struct OrderSlot {
    order_id:     String,
    symbol:       String,
    side:         i8,    // 1=BUY, -1=SELL
    price:        f64,
    size:         f64,
    ts:           f64,
    qty_in_front: f64,
}

// (order_id, symbol, side, price, size, is_maker, ts, qty_in_front, queue_displacement_us)
type FillTuple = (String, String, i8, f64, f64, bool, f64, f64, f64);

#[pyclass]
pub struct FIFOQueueCore {
    slots:        HashMap<String, OrderSlot>,
    prev_book:    HashMap<u64, f64>,  // f64.to_bits() as key — safe for finite prices
    cancel_ratio: f64,
}

#[pymethods]
impl FIFOQueueCore {
    #[new]
    #[pyo3(signature = (cancel_ratio = 0.20))]
    fn new(cancel_ratio: f64) -> Self {
        FIFOQueueCore {
            slots: HashMap::new(),
            prev_book: HashMap::new(),
            cancel_ratio,
        }
    }

    // qty_in_front is computed Python-side (_qty_in_front needs the full book)
    fn register(
        &mut self,
        order_id: &str, symbol: &str,
        side: i8, price: f64, size: f64, ts: f64,
        qty_in_front: f64,
    ) {
        self.slots.insert(order_id.to_string(), OrderSlot {
            order_id: order_id.to_string(),
            symbol:   symbol.to_string(),
            side, price, size, ts, qty_in_front,
        });
    }

    fn cancel(&mut self, order_id: &str) -> bool {
        self.slots.remove(order_id).is_some()
    }

    fn process_tick(
        &mut self,
        bids:   Vec<(f64, f64)>,
        asks:   Vec<(f64, f64)>,
        trades: Vec<(f64, f64, i8, f64)>,
    ) -> Vec<FillTuple> {
        if self.slots.is_empty() {
            self.prev_book = to_book_map(&bids, &asks);
            return Vec::new();
        }

        let current = to_book_map(&bids, &asks);
        self.infer_cancels(&current);

        let mut fills:      Vec<FillTuple> = Vec::new();
        let mut filled_ids: Vec<String>    = Vec::new();

        for (tp, ts, trade_side, trade_ts) in &trades {
            for (oid, slot) in self.slots.iter_mut() {
                if filled_ids.contains(oid) {
                    continue;
                }
                if let Some(f) = try_fill(slot, *tp, *ts, *trade_side, *trade_ts, self.cancel_ratio) {
                    fills.push(f);
                    filled_ids.push(oid.clone());
                }
            }
        }

        for oid in &filled_ids {
            self.slots.remove(oid);
        }

        self.prev_book = current;
        fills
    }

    fn pending_count(&self) -> usize {
        self.slots.len()
    }

    fn active_order_ids(&self) -> Vec<String> {
        self.slots.keys().cloned().collect()
    }
}

impl FIFOQueueCore {
    // two-pass: collect deltas (immutable), then apply (mutable)
    // borrow checker, you know how it is
    fn infer_cancels(&mut self, current: &HashMap<u64, f64>) {
        let deltas: Vec<(String, f64)> = self.slots
            .iter()
            .filter_map(|(oid, slot)| {
                let pb   = slot.price.to_bits();
                let prev = self.prev_book.get(&pb).copied().unwrap_or(0.0);
                let curr = current.get(&pb).copied().unwrap_or(0.0);
                if prev > curr + 1e-12 { Some((oid.clone(), prev - curr)) } else { None }
            })
            .collect();

        for (oid, cancelled) in deltas {
            if let Some(slot) = self.slots.get_mut(&oid) {
                slot.qty_in_front = (slot.qty_in_front - cancelled).max(0.0);
            }
        }
    }
}

fn to_book_map(bids: &[(f64, f64)], asks: &[(f64, f64)]) -> HashMap<u64, f64> {
    bids.iter().chain(asks.iter()).map(|(p, s)| (p.to_bits(), *s)).collect()
}

fn try_fill(
    slot:         &mut OrderSlot,
    trade_price:  f64,
    trade_size:   f64,
    trade_side:   i8,
    trade_ts:     f64,
    cancel_ratio: f64,
) -> Option<FillTuple> {
    match slot.side {
        1  => { if trade_side != -1 || trade_price > slot.price + 1e-12 { return None; } }
        -1 => { if trade_side !=  1 || trade_price < slot.price - 1e-12 { return None; } }
        _  => return None,
    }

    let fill_size = if slot.qty_in_front > 0.0 {
        let consumed  = trade_size * (1.0 + cancel_ratio);
        let raw_q     = slot.qty_in_front - consumed;
        let overshoot = (-raw_q).max(0.0);
        slot.qty_in_front = raw_q.max(0.0);
        if overshoot <= 1e-12 { return None; }
        slot.size.min(overshoot)
    } else {
        slot.size.min(trade_size)
    };

    if fill_size <= 1e-12 {
        return None;
    }

    Some((
        slot.order_id.clone(), slot.symbol.clone(),
        slot.side, slot.price, fill_size,
        true,              // is_maker
        trade_ts,
        slot.qty_in_front,
        0.0,               // queue_displacement_us, set by engine layer
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FIFOQueueCore>()?;
    Ok(())
}
