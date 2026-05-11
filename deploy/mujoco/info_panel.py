"""Pygame-based info panel displaying force and stiffness data alongside MuJoCo."""

import os
import pygame
import config


# Shared state — written by PhysicsViewerThread, read by InfoPanelThread
panel_state = {
    'force_mag': 0.0, 'fx': 0.0, 'fy': 0.0, 'fz': 0.0,
    'stiffness': float('nan'), 'kd': float('nan'),
}


def run(viewer):
    """Run the info panel. Blocks until the MuJoCo viewer closes.

    Pass the mujoco passive-viewer handle so the panel exits when the sim does.
    """
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,920")
    pygame.init()

    W, H = 500, 240
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Robot Status")

    # ── Colour palette ────────────────────────────────────────────────────────
    BG        = (15, 18, 28)
    DIV       = (35, 40, 58)
    SEC_COL   = (110, 120, 150)
    MAG_COL   = (170, 200, 255)
    FX_COL    = (255, 130, 110)
    FY_COL    = (110, 230, 145)
    FZ_COL    = (110, 175, 255)
    KP_COL    = (90, 235, 135)
    KD_COL    = (170, 175, 200)
    BAR_BG    = (38, 44, 62)
    BAR_FILL  = (75, 195, 115)
    HINT_COL  = (85, 92, 118)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    font_sec   = pygame.font.SysFont("dejavusans", 12, bold=True)
    font_val   = pygame.font.SysFont("dejavusans", 14)
    font_large = pygame.font.SysFont("dejavusans", 24, bold=True)
    font_hint  = pygame.font.SysFont("dejavusans", 12)

    # ── Progress bar geometry ─────────────────────────────────────────────────
    BAR_X, BAR_Y, BAR_W, BAR_H = 168, 152, 190, 16
    STIFF_ZERO, STIFF_FULL = 400.0, 1500.0

    clock = pygame.time.Clock()

    while viewer.is_running():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        s         = panel_state
        stiffness = s['stiffness']
        kd        = s['kd']
        force_mag = s['force_mag']
        fx, fy, fz = s['fx'], s['fy'], s['fz']
        stiff_ok  = (stiffness == stiffness)  # False when NaN
        kd_ok     = (kd == kd)

        screen.fill(BG)

        # ── Forces ────────────────────────────────────────────────────────────
        screen.blit(font_sec.render("FORCES", True, SEC_COL), (15, 10))
        screen.blit(font_large.render(f"|F|  {force_mag:.1f} N", True, MAG_COL), (15, 26))

        col_x = 15
        for label, val, col in (("Fx", fx, FX_COL), ("Fy", fy, FY_COL), ("Fz", fz, FZ_COL)):
            screen.blit(font_sec.render(label, True, SEC_COL), (col_x, 64))
            screen.blit(font_val.render(f"{val:+.1f}", True, col), (col_x, 78))
            col_x += 155

        pygame.draw.line(screen, DIV, (15, 108), (W - 15, 108), 1)

        # ── Stiffness ─────────────────────────────────────────────────────────
        screen.blit(font_sec.render("STIFFNESS", True, SEC_COL), (15, 118))

        kp_str = f"{stiffness:.0f}" if stiff_ok else "n/a"
        screen.blit(font_large.render(f"kp  {kp_str}", True, KP_COL), (15, 132))

        pygame.draw.rect(screen, BAR_BG, (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
        if stiff_ok:
            fill_w = int(max(0, min(BAR_W, (stiffness - STIFF_ZERO) / (STIFF_FULL - STIFF_ZERO) * BAR_W)))
            if fill_w > 0:
                pygame.draw.rect(screen, BAR_FILL, (BAR_X, BAR_Y, fill_w, BAR_H), border_radius=4)

        kd_label = f"kd  {kd:.3f}" if kd_ok else "kd  n/a"
        screen.blit(font_val.render(kd_label, True, KD_COL), (BAR_X + BAR_W + 12, BAR_Y))

        # ── Key hint ──────────────────────────────────────────────────────────
        if config.ENABLE_STIFFNESS_KEYS:
            hint = f"[ = less           ] = more           ±{config.STIFFNESS_KEY_STEP:.0f}"
            screen.blit(font_hint.render(hint, True, HINT_COL), (15, 200))

        pygame.display.flip()
        clock.tick(50)

    pygame.quit()
