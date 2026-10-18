package tui

import (
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/lingduoduo/Agentic-Search/cli/config"
)

func TestModifiedKeysRetainChatShortcuts(t *testing.T) {
	m := NewModel(config.Config{}, nil)
	m.initInputReady = true
	updated, _ := m.Update(tea.KeyPressMsg{Code: 'o', Mod: tea.ModCtrl})
	m = updated.(Model)
	if !m.viewport.showSources {
		t.Fatal("Ctrl+O should toggle sources")
	}
	updated, _ = m.Update(tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl})
	if !updated.(Model).quitPending {
		t.Fatal("Ctrl+D should request quit confirmation")
	}
}

func TestServeViewEnablesTerminalModesAfterAuthentication(t *testing.T) {
	m := NewServeModel(config.Config{}, "", nil)
	if view := m.View(); view.AltScreen || view.MouseMode != tea.MouseModeNone {
		t.Fatal("authentication should render in the main screen without mouse reporting")
	}
	m.authed = true
	m.tui = NewModel(config.Config{}, nil)
	if view := m.View(); !view.AltScreen || view.MouseMode != tea.MouseModeCellMotion {
		t.Fatal("chat should enable the alternate screen and mouse reporting")
	}
}

func TestAuthControlKeysAbort(t *testing.T) {
	for _, code := range []rune{'c', 'd'} {
		m := NewAuthModel("", "", nil)
		m, _ = m.Update(tea.KeyPressMsg{Code: code, Mod: tea.ModCtrl})
		if !m.Aborted {
			t.Errorf("Ctrl+%c should abort authentication", code)
		}
	}
}
