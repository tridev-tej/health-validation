// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package xid

import (
	"strings"
	"sync"

	"github.com/dsx-ai-factory/health-validation/catalog/health"
)

// ScopeObservedValues implements [health.ObservedValuesScoper]: a check
// reports only its own events, grouped, and count entries. A check's own are
// the codes it declares in ErrorCodes; gpu_xid_unknown's are the codes no
// check covers, and gpu_xid_informational's are the informational codes no
// check covers.
func (m XidMeasurement) ScopeObservedValues(checkID string, observed map[string]any) map[string]any {
	switch checkID {
	case unknownCheckID:
		return checkScope{rest: m.isUnknown}.apply(observed)
	case informationalCheckID:
		return checkScope{rest: m.isInformational}.apply(observed)
	}
	scopes := m.scopes
	if scopes == nil {
		scopes = builtinScopes()
	}
	s, ok := scopes[checkID]
	if !ok {
		// shortcut: a check with no ErrorCodes reports every code; ask authors to declare theirs.
		return observed
	}
	return s.apply(observed)
}

// checkScope is what one check judges: Xid codes, NVLink5 fault subtypes
// such as "149.4" (see [XidEvent.SubXidCode]), or, for the catch-all checks,
// every code that satisfies rest.
type checkScope struct {
	codes map[int64]bool
	subs  map[string]bool
	rest  func(code int64) bool // when set, the scope is the codes it accepts
}

// scopesFrom maps each Xid check in defs that declares ErrorCodes to its scope.
func scopesFrom(defs []*health.CheckDefinition) map[string]checkScope {
	out := map[string]checkScope{}
	for _, def := range defs {
		if def.Category != health.Xid || len(def.ErrorCodes) == 0 {
			continue
		}
		s := checkScope{codes: map[int64]bool{}, subs: map[string]bool{}}
		for _, c := range def.ErrorCodes {
			if code, ok := parseCode(c); ok {
				s.codes[code] = true
			} else if strings.Contains(c, ".") {
				s.subs[c] = true
			}
		}
		out[def.ID] = s
	}
	return out
}

// builtinScopes is scopesFrom for the checks [Checks] builds.
var builtinScopes = sync.OnceValue(func() map[string]checkScope { return scopesFrom(builtinDefs()) })

// keepCode reports whether counts for code belong to the check.
func (s checkScope) keepCode(code int64) bool {
	if s.rest != nil {
		return s.rest(code)
	}
	return s.codes[code]
}

// keep reports whether e is one of the check's own events. A sub-code
// keeps the events sub_xid_code_counts counts under its subtype.
func (s checkScope) keep(e XidEvent) bool {
	if s.keepCode(e.XidCode) {
		return true
	}
	if len(s.subs) == 0 {
		return false
	}
	sub, ok := e.SubXidCode()
	return ok && s.subs[sub] && hasSubXidCodeChecks(e.XidCode, e.NVLink5.SubComponent)
}

// apply returns a narrowed copy, because observed is shared with the rules.
// A scope with no codes leaves the code-keyed counts alone, and one with no
// subtypes leaves the sub-code counts alone.
func (s checkScope) apply(observed map[string]any) map[string]any {
	out := make(map[string]any, len(observed))
	for name, value := range observed {
		switch v := value.(type) {
		case []XidEvent:
			if name == SignalEvents {
				value = groupEvents(v, s.keep)
			}
		case map[int64]int64:
			if (len(s.codes) > 0 || s.rest != nil) && (name == SignalXidCounts || name == SignalFatalXidCounts || name == SignalUnknownXidCounts || name == SignalInformationalXidCounts) {
				value = onlyKeys(v, s.keepCode)
			}
		case map[string]int64:
			if len(s.subs) > 0 && name == SignalSubXidCodeCounts {
				value = onlyKeys(v, func(sub string) bool { return s.subs[sub] })
			}
		}
		out[name] = value
	}
	return out
}

// groupEvents merges events with the same message, since every field is
// parsed from the line, and counts them in Count.
func groupEvents(events []XidEvent, keep func(XidEvent) bool) []XidEvent {
	var out []XidEvent
	index := map[string]int{} // message -> its event in out
	for _, e := range events {
		if !keep(e) {
			continue
		}
		if i, ok := index[e.Message]; ok {
			out[i].Count++
			continue
		}
		e.Count = 1
		// shortcut: events without a message are never merged; group by the other fields if outside producers omit it.
		if e.Message != "" {
			index[e.Message] = len(out)
		}
		out = append(out, e)
	}
	return out
}

// onlyKeys returns nil when no key is kept, as a measurement with no events does.
func onlyKeys[K comparable](counts map[K]int64, keep func(K) bool) map[K]int64 {
	var out map[K]int64
	for k, n := range counts {
		if keep(k) {
			if out == nil {
				out = map[K]int64{}
			}
			out[k] = n
		}
	}
	return out
}
