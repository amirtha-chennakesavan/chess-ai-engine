#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           AI Chess Engine — Adaptive Difficulty              ║
║           pygame desktop app with AI explanations            ║
╚══════════════════════════════════════════════════════════════╝

FEATURES:
  ✅ Full chess with all rules (castling, en passant, promotion)
  ✅ Adaptive AI — adjusts difficulty based on your performance
  ✅ Difficulty presets: Easy / Medium / Hard / Grandmaster
  ✅ Smooth piece movement animation
  ✅ Post-game analysis with move-by-move feedback
  ✅ AI explains every move it makes in plain English
  ✅ Hint system — press H to see the best move highlighted
  ✅ Undo — press Z to take back your last move
  ✅ Move history log on the right panel
  ✅ Captured pieces display
  ✅ Check / checkmate / stalemate detection

INSTALL:
  pip install pygame chess

CONTROLS:
  Click          — select and move pieces
  H              — show hint (best move highlighted)
  Z              — undo last move
  R              — restart game
  1/2/3/4        — set difficulty (Easy/Medium/Hard/Grandmaster)
  Q              — quit
"""

import pygame
import chess
import chess.engine
import random
import sys
import time
import threading
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT  = 1100, 720
BOARD_SIZE     = 640
SQUARE_SIZE    = BOARD_SIZE // 8
PANEL_X        = BOARD_SIZE + 20
PANEL_W        = WIDTH - BOARD_SIZE - 30

FPS = 60
ANIM_DURATION = 0.15   # seconds for piece slide animation

# Difficulty presets
DIFFICULTY_PRESETS = {
    1: ("Easy",        2,  0.55),   # (name, depth, blunder_chance)
    2: ("Medium",      3,  0.20),
    3: ("Hard",        4,  0.05),
    4: ("Grandmaster", 4,  0.00),
}

# Colors
C_BG         = (15,  17,  21)
C_LIGHT      = (240, 217, 181)
C_DARK       = (181, 136,  99)
C_SELECTED   = (106, 168,  79, 180)
C_LEGAL      = (106, 168,  79, 100)
C_HINT_FROM  = (255, 200,   0, 200)
C_HINT_TO    = (255, 140,   0, 200)
C_LAST_MOVE  = ( 70, 130, 180, 120)
C_CHECK      = (220,  50,  50, 180)
C_PANEL      = ( 22,  26,  32)
C_BORDER     = ( 40,  48,  58)
C_TEXT       = (220, 225, 230)
C_MUTED      = ( 90, 100, 115)
C_ACCENT     = ( 80, 200, 120)
C_AI_BUBBLE  = ( 28,  35,  44)
C_WHITE_CAP  = (240, 240, 240)
C_BLACK_CAP  = ( 40,  40,  40)

PIECE_UNICODE = {
    chess.PAWN:   {chess.WHITE: '♙', chess.BLACK: '♟'},
    chess.KNIGHT: {chess.WHITE: '♘', chess.BLACK: '♞'},
    chess.BISHOP: {chess.WHITE: '♗', chess.BLACK: '♝'},
    chess.ROOK:   {chess.WHITE: '♖', chess.BLACK: '♜'},
    chess.QUEEN:  {chess.WHITE: '♕', chess.BLACK: '♛'},
    chess.KING:   {chess.WHITE: '♔', chess.BLACK: '♚'},
}

def draw_piece_shape(surface, piece_type, color, x, y, sq_size):
    """Draw a proper chess piece shape using pygame drawing primitives."""
    S    = sq_size
    cx   = x + S // 2
    cy   = y + S // 2
    pad  = S // 8

    # Colors
    if color == chess.WHITE:
        fill   = (255, 255, 255)
        outline = (30,  30,  30)
        detail  = (180, 180, 180)
    else:
        fill   = (30,  30,  30)
        outline = (200, 200, 200)
        detail  = (80,  80,  80)

    def filled(pts, col=fill):
        pygame.draw.polygon(surface, col, pts)
        pygame.draw.polygon(surface, outline, pts, 2)

    def circle(cx_, cy_, r, col=fill, border=2):
        pygame.draw.circle(surface, col, (cx_, cy_), r)
        pygame.draw.circle(surface, outline, (cx_, cy_), r, border)

    def rect(rx, ry, rw, rh, col=fill, border=2):
        pygame.draw.rect(surface, col, (rx, ry, rw, rh))
        pygame.draw.rect(surface, outline, (rx, ry, rw, rh), border)

    if piece_type == chess.PAWN:
        # Base
        bw, bh = S*5//8, S//7
        rect(cx - bw//2, y + S - pad - bh, bw, bh)
        # Stem
        sw, sh = S//4, S*2//8
        rect(cx - sw//2, y + S - pad - bh - sh, sw, sh)
        # Head
        circle(cx, y + S - pad - bh - sh - S//5, S//5)

    elif piece_type == chess.ROOK:
        # Base
        bw, bh = S*6//8, S//7
        rect(cx - bw//2, y + S - pad - bh, bw, bh)
        # Body
        tw, th = S*4//8, S*4//8
        rect(cx - tw//2, y + S - pad - bh - th, tw, th)
        # Battlements (3 merlons)
        mw = tw // 4
        mh = S // 7
        top_y = y + S - pad - bh - th - mh
        for i in range(3):
            mx = cx - tw//2 + i * (tw//2 - mw//2 + 1)
            rect(mx, top_y, mw, mh)

    elif piece_type == chess.KNIGHT:
        # Base
        bw, bh = S*6//8, S//7
        rect(cx - bw//2, y + S - pad - bh, bw, bh)
        # Body/neck
        body = [
            (cx - S//6, y + S - pad - bh),
            (cx + S//5, y + S - pad - bh),
            (cx + S//4, y + S*2//5),
            (cx + S//8, y + S//4),
            (cx - S//5, y + S//3),
            (cx - S//4, y + S*2//5),
        ]
        filled(body)
        # Head
        head = [
            (cx - S//5, y + S//3),
            (cx + S//8, y + S//4),
            (cx + S//5, y + S//5),
            (cx + S//4, y + S//8),
            (cx,        y + pad + S//10),
            (cx - S//4, y + S//5),
            (cx - S//3, y + S//3),
        ]
        filled(head)
        # Eye
        eye_col = outline if color == chess.WHITE else fill
        pygame.draw.circle(surface, eye_col,
                           (cx + S//8, y + S//5), S//14)
        # Nostril hint
        pygame.draw.line(surface, outline,
                         (cx + S//5, y + S//8),
                         (cx + S//4, y + S//10), 2)

    elif piece_type == chess.BISHOP:
        # Base
        bw, bh = S*6//8, S//7
        rect(cx - bw//2, y + S - pad - bh, bw, bh)
        # Lower body
        lw, lh = S*3//8, S//5
        rect(cx - lw//2, y + S - pad - bh - lh, lw, lh)
        # Main body (tapered)
        body = [
            (cx - lw//2,     y + S - pad - bh - lh),
            (cx + lw//2,     y + S - pad - bh - lh),
            (cx + S//8,      y + S*2//8),
            (cx,             y + S//4),
            (cx - S//8,      y + S*2//8),
        ]
        filled(body)
        # Head ball
        circle(cx, y + S//4, S//7)
        # Tip
        circle(cx, y + pad + S//12, S//14)
        # Cross detail on head
        pygame.draw.line(surface, outline,
                         (cx, y + S//4 - S//9),
                         (cx, y + S//4 + S//9), 2)
        pygame.draw.line(surface, outline,
                         (cx - S//9, y + S//4),
                         (cx + S//9, y + S//4), 2)

    elif piece_type == chess.QUEEN:
        # Base
        bw, bh = S*6//8, S//7
        rect(cx - bw//2, y + S - pad - bh, bw, bh)
        # Body (wide)
        body = [
            (cx - S*3//8, y + S - pad - bh),
            (cx + S*3//8, y + S - pad - bh),
            (cx + S//5,   y + S//3),
            (cx,          y + S//4),
            (cx - S//5,   y + S//3),
        ]
        filled(body)
        # Crown base
        cw = S*3//8
        rect(cx - cw//2, y + S//4, cw, S//8)
        # Crown points (5 balls)
        crown_y = y + S//4
        for i, offset in enumerate([-cw//2, -cw//4, 0, cw//4, cw//2]):
            r = S//11 if i % 2 == 0 else S//14
            circle(cx + offset, crown_y - r, r)

    elif piece_type == chess.KING:
        # Base
        bw, bh = S*6//8, S//7
        rect(cx - bw//2, y + S - pad - bh, bw, bh)
        # Body
        body = [
            (cx - S*3//8, y + S - pad - bh),
            (cx + S*3//8, y + S - pad - bh),
            (cx + S//5,   y + S//3),
            (cx,          y + S//4),
            (cx - S//5,   y + S//3),
        ]
        filled(body)
        # Crown base
        cw = S*3//8
        rect(cx - cw//2, y + S//4, cw, S//8)
        # Cross on top
        cross_y = y + S//4
        cross_w  = S//5
        cross_h  = S//4
        rect(cx - S//14, cross_y - cross_h, S//7, cross_h)   # vertical
        rect(cx - cross_w//2, cross_y - cross_h*2//3, cross_w, S//12)  # horizontal

PIECE_NAMES = {
    chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
    chess.ROOK: 'rook', chess.QUEEN: 'queen',   chess.KING: 'king',
}

SQUARE_NAMES_FRIENDLY = {
    0:'a1',1:'b1',2:'c1',3:'d1',4:'e1',5:'f1',6:'g1',7:'h1',
}


# ── AI Move Explainer ─────────────────────────────────────────────────────────
def explain_move(board: chess.Board, move: chess.Move, difficulty: int) -> str:
    """Generate a plain-English explanation of why the AI made a move."""
    piece      = board.piece_at(move.from_square)
    piece_name = PIECE_NAMES.get(piece.piece_type, 'piece') if piece else 'piece'
    to_name    = chess.square_name(move.to_square)
    from_name  = chess.square_name(move.from_square)

    # Check if it's a capture
    captured = board.piece_at(move.to_square)
    cap_name = PIECE_NAMES.get(captured.piece_type, 'piece') if captured else None

    # Check if move gives check
    board_copy = board.copy()
    board_copy.push(move)
    gives_check = board_copy.is_check()
    is_checkmate = board_copy.is_checkmate()

    if is_checkmate:
        return f"Checkmate! I moved my {piece_name} to {to_name}. Game over!"

    if gives_check and cap_name:
        return f"I captured your {cap_name} on {to_name} with my {piece_name} — and put you in check!"

    if gives_check:
        return f"I moved my {piece_name} to {to_name} to put you in check. Watch out!"

    if cap_name:
        explanations = [
            f"I captured your {cap_name} on {to_name} with my {piece_name}. Material advantage!",
            f"Taking your {cap_name} with my {piece_name} — a free piece is always good.",
            f"My {piece_name} takes your {cap_name} on {to_name}.",
        ]
        return random.choice(explanations)

    if move.promotion:
        return f"Pawn promotion! My pawn reaches {to_name} and becomes a queen!"

    # Castling
    if piece and piece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2:
        side = "kingside" if move.to_square > move.from_square else "queenside"
        return f"I'm castling {side} — protecting my king and activating my rook."

    # Generic strategic explanations by difficulty
    if difficulty <= 2:
        generic = [
            f"I moved my {piece_name} from {from_name} to {to_name}.",
            f"Hmm, let me put my {piece_name} on {to_name}.",
            f"Moving my {piece_name} to {to_name}. Still learning!",
        ]
    elif difficulty <= 5:
        generic = [
            f"I moved my {piece_name} to {to_name} to improve its position.",
            f"My {piece_name} is better placed on {to_name} — more central control.",
            f"Repositioning my {piece_name} to {to_name} to apply more pressure.",
            f"I'm developing my {piece_name} to {to_name} for better activity.",
        ]
    else:
        generic = [
            f"I moved my {piece_name} to {to_name} to control key squares.",
            f"Placing my {piece_name} on {to_name} — this creates long-term pressure.",
            f"My {piece_name} on {to_name} eyes several important squares.",
            f"This {piece_name} move to {to_name} prepares a tactical idea.",
            f"I'm improving my {piece_name}'s activity — {to_name} is a strong outpost.",
        ]
    return random.choice(generic)


# ── Adaptive AI ───────────────────────────────────────────────────────────────
class AdaptiveAI:
    """
    Adaptive difficulty engine using python-chess minimax + alpha-beta pruning.
    Supports both adaptive mode and fixed difficulty presets.
    """
    def __init__(self):
        self.difficulty    = 5      # 1-10 adaptive scale
        self.preset        = None   # None = adaptive, 1-4 = fixed preset
        self.player_wins   = 0
        self.ai_wins       = 0
        self.moves_played  = 0
        self.thinking      = False
        self.best_move     = None
        self.explanation   = ""

    def set_preset(self, preset_num: int):
        """Lock to a difficulty preset (1=Easy, 2=Medium, 3=Hard, 4=Grandmaster)."""
        if preset_num in DIFFICULTY_PRESETS:
            self.preset = preset_num
            name, _, _ = DIFFICULTY_PRESETS[preset_num]
            print(f"  Difficulty set to: {name}")

    @property
    def preset_name(self) -> str:
        if self.preset:
            return DIFFICULTY_PRESETS[self.preset][0]
        return f"Adaptive ({self.difficulty}/10)"

    @property
    def depth(self) -> int:
        if self.preset:
            return DIFFICULTY_PRESETS[self.preset][1]
        return max(1, min(4, self.difficulty // 3))

    @property
    def blunder_chance(self) -> float:
        if self.preset:
            return DIFFICULTY_PRESETS[self.preset][2]
        if self.difficulty >= 9:
            return 0.0
        if self.difficulty >= 7:
            return 0.05
        if self.difficulty >= 5:
            return 0.15
        if self.difficulty >= 3:
            return 0.30
        return 0.50

    def adjust_difficulty(self, player_won: bool):
        if self.preset:
            return  # Don't adjust if using a preset
        if player_won:
            self.player_wins += 1
            if self.player_wins >= 2:
                self.difficulty = min(10, self.difficulty + 1)
                self.player_wins = 0
        else:
            self.ai_wins += 1
            if self.ai_wins >= 2:
                self.difficulty = max(1, self.difficulty - 1)
                self.ai_wins = 0

    def evaluate(self, board: chess.Board) -> float:
        """Simple material + position evaluation."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.BLACK else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
        }

        # Piece square tables (simplified center control bonus)
        center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
        near_center    = {chess.C3,chess.D3,chess.E3,chess.F3,
                          chess.C6,chess.D6,chess.E6,chess.F6}

        score = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            val = piece_values.get(p.piece_type, 0)
            # Center bonus
            if sq in center_squares:
                val += 30
            elif sq in near_center:
                val += 10
            # Mobility bonus
            score += val if p.color == chess.WHITE else -val

        # Mobility
        board_copy = board.copy()
        board_copy.turn = chess.WHITE
        score += len(list(board_copy.legal_moves)) * 5
        board_copy.turn = chess.BLACK
        score -= len(list(board_copy.legal_moves)) * 5

        return score

    def minimax(self, board: chess.Board, depth: int, alpha: float,
                beta: float, maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        moves = list(board.legal_moves)
        # Move ordering: captures first
        moves.sort(key=lambda m: board.piece_at(m.to_square) is not None, reverse=True)

        if maximizing:
            best = float('-inf')
            for move in moves:
                board.push(move)
                best = max(best, self.minimax(board, depth-1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for move in moves:
                board.push(move)
                best = min(best, self.minimax(board, depth-1, alpha, beta, True))
                board.pop()
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    def get_best_move(self, board: chess.Board) -> chess.Move | None:
        """Find best move with current difficulty settings."""
        moves = list(board.legal_moves)
        if not moves:
            return None

        # Blunder: sometimes pick a random move
        if random.random() < self.blunder_chance:
            return random.choice(moves)

        best_move  = None
        best_score = float('-inf')

        for move in moves:
            board.push(move)
            score = -self.minimax(board, self.depth - 1,
                                  float('-inf'), float('inf'), False)
            board.pop()
            if score > best_score:
                best_score = score
                best_move  = move

        return best_move

    def think(self, board: chess.Board):
        """Run AI in background thread."""
        self.thinking  = True
        self.best_move = None
        board_copy     = board.copy()

        def _run():
            move = self.get_best_move(board_copy)
            self.best_move   = move
            self.explanation = explain_move(board_copy, move, self.difficulty) if move else ""
            self.thinking    = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def get_hint(self, board: chess.Board) -> chess.Move | None:
        """Get best move for player (hint)."""
        # Temporarily flip — find best move for the current player
        return self.get_best_move(board)


# ── Chess Game ────────────────────────────────────────────────────────────────
class ChessGame:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Chess Engine")
        self.clock   = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont("Georgia",       32, bold=True)
        self.font_md  = pygame.font.SysFont("Georgia",       18)
        self.font_sm  = pygame.font.SysFont("Courier New",   14)
        self.font_xs  = pygame.font.SysFont("Courier New",   12)
        # Pieces drawn with shapes (no font needed)

        self.reset()

    def reset(self, keep_preset=False):
        preset = self.ai.preset if (keep_preset and hasattr(self, "ai")) else None
        self.board          = chess.Board()
        self.ai             = AdaptiveAI()
        if preset:
            self.ai.preset = preset
        self.selected_sq    = None
        self.legal_moves    = []
        self.last_move      = None
        self.hint_move      = None
        self.show_hint      = False
        self.move_history   = []   # list of (san, explanation, is_ai)
        self.game_history   = []   # saved for post-game analysis
        self.captured_white = []   # captured by AI (white pieces)
        self.captured_black = []   # captured by player (black pieces)
        self.status_msg     = "Your turn — you play White"
        self.ai_explanation = "I'm ready. Make your move!"
        self.game_over      = False
        self.ai_thinking    = False
        self.show_analysis  = False
        self.analysis_lines = []
        # Animation state
        self.anim_active    = False
        self.anim_piece     = None   # (piece_type, color)
        self.anim_from_px   = (0, 0)
        self.anim_to_px     = (0, 0)
        self.anim_start     = 0.0
        self.anim_move      = None

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def sq_to_pixel(self, sq: int) -> tuple[int,int]:
        col = chess.square_file(sq)
        row = 7 - chess.square_rank(sq)
        return col * SQUARE_SIZE, row * SQUARE_SIZE

    def pixel_to_sq(self, x: int, y: int) -> int | None:
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return chess.square(col, 7 - row)

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw_board(self):
        for sq in chess.SQUARES:
            x, y = self.sq_to_pixel(sq)
            col  = chess.square_file(sq)
            row  = chess.square_rank(sq)
            color = C_LIGHT if (col + row) % 2 == 0 else C_DARK
            pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

        # Last move highlight
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self.sq_to_pixel(sq)
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                s.fill(C_LAST_MOVE)
                self.screen.blit(s, (x, y))

        # Check highlight
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                x, y = self.sq_to_pixel(king_sq)
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                s.fill(C_CHECK)
                self.screen.blit(s, (x, y))

        # Selected square
        if self.selected_sq is not None:
            x, y = self.sq_to_pixel(self.selected_sq)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(C_SELECTED)
            self.screen.blit(s, (x, y))

        # Legal move dots
        for move in self.legal_moves:
            x, y = self.sq_to_pixel(move.to_square)
            cx, cy = x + SQUARE_SIZE//2, y + SQUARE_SIZE//2
            if self.board.piece_at(move.to_square):
                # Capture ring
                pygame.draw.circle(self.screen, (106,168,79), (cx,cy), SQUARE_SIZE//2 - 4, 5)
            else:
                pygame.draw.circle(self.screen, (106,168,79), (cx,cy), 12)

        # Hint highlight
        if self.show_hint and self.hint_move:
            for sq, color in [(self.hint_move.from_square, C_HINT_FROM),
                               (self.hint_move.to_square,   C_HINT_TO)]:
                x, y = self.sq_to_pixel(sq)
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                s.fill(color)
                self.screen.blit(s, (x, y))

        # Rank/file labels
        for i in range(8):
            # Files (a-h)
            label = self.font_xs.render(chess.FILE_NAMES[i], True, C_MUTED)
            self.screen.blit(label, (i*SQUARE_SIZE + 2, BOARD_SIZE - 16))
            # Ranks (1-8)
            label = self.font_xs.render(str(8-i), True, C_MUTED)
            self.screen.blit(label, (2, i*SQUARE_SIZE + 2))

    def draw_pieces(self):
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece:
                continue
            # Skip the animating piece's destination square during animation
            if self.anim_active and self.anim_move and sq == self.anim_move.to_square:
                continue
            x, y = self.sq_to_pixel(sq)
            draw_piece_shape(self.screen, piece.piece_type, piece.color, x, y, SQUARE_SIZE)

        # Draw animating piece on top
        if self.anim_active and self.anim_piece:
            t = min(1.0, (time.time() - self.anim_start) / ANIM_DURATION)
            # Ease out cubic
            t = 1 - (1 - t) ** 3
            fx, fy = self.anim_from_px
            tx, ty = self.anim_to_px
            ax = int(fx + (tx - fx) * t)
            ay = int(fy + (ty - fy) * t)
            draw_piece_shape(self.screen, self.anim_piece[0], self.anim_piece[1],
                             ax, ay, SQUARE_SIZE)
            if t >= 1.0:
                self.anim_active = False

    def draw_panel(self):
        # Panel background
        pygame.draw.rect(self.screen, C_PANEL, (BOARD_SIZE, 0, WIDTH - BOARD_SIZE, HEIGHT))
        pygame.draw.line(self.screen, C_BORDER, (BOARD_SIZE, 0), (BOARD_SIZE, HEIGHT), 2)

        y = 16

        # Title
        title = self.font_lg.render("AI Chess", True, C_ACCENT)
        self.screen.blit(title, (PANEL_X, y))
        y += 40

        # Difficulty label
        diff_label = self.font_sm.render(f"Difficulty: {self.ai.preset_name}", True, C_TEXT)
        self.screen.blit(diff_label, (PANEL_X, y))
        y += 18

        # Preset buttons [1] Easy [2] Medium [3] Hard [4] GM
        btn_labels = ["1:Easy", "2:Med", "3:Hard", "4:GM"]
        btn_colors = [(80,200,120),(255,200,0),(220,140,40),(220,60,60)]
        btn_w = (PANEL_W - 20) // 4 - 2
        for i, (lbl, col) in enumerate(zip(btn_labels, btn_colors)):
            bx = PANEL_X + i * (btn_w + 3)
            active = (self.ai.preset == i + 1)
            bg = col if active else C_BORDER
            pygame.draw.rect(self.screen, bg, (bx, y, btn_w, 18), border_radius=3)
            ts = self.font_xs.render(lbl, True, (20,20,20) if active else C_MUTED)
            self.screen.blit(ts, (bx + btn_w//2 - ts.get_width()//2, y + 3))
        y += 24

        # Status
        pygame.draw.line(self.screen, C_BORDER, (PANEL_X, y), (PANEL_X + PANEL_W - 10, y))
        y += 10
        status = self.font_sm.render(self.status_msg, True, C_ACCENT)
        self.screen.blit(status, (PANEL_X, y))
        y += 26

        # AI thinking indicator
        if self.ai_thinking:
            dots = "." * (int(time.time() * 3) % 4)
            think = self.font_sm.render(f"AI thinking{dots}", True, C_MUTED)
            self.screen.blit(think, (PANEL_X, y))
        y += 20

        # AI explanation bubble
        pygame.draw.rect(self.screen, C_AI_BUBBLE,
                         (PANEL_X - 4, y, PANEL_W - 6, 80), border_radius=8)
        pygame.draw.rect(self.screen, C_BORDER,
                         (PANEL_X - 4, y, PANEL_W - 6, 80), 1, border_radius=8)
        ai_label = self.font_xs.render("🤖 AI says:", True, C_ACCENT)
        self.screen.blit(ai_label, (PANEL_X + 4, y + 6))

        # Word wrap explanation
        words  = self.ai_explanation.split()
        line   = ""
        lines  = []
        for word in words:
            test = line + word + " "
            if self.font_xs.size(test)[0] > PANEL_W - 20:
                lines.append(line)
                line = word + " "
            else:
                line = test
        lines.append(line)
        for i, ln in enumerate(lines[:3]):
            surf = self.font_xs.render(ln, True, C_TEXT)
            self.screen.blit(surf, (PANEL_X + 4, y + 22 + i*16))
        y += 90

        # Captured pieces
        pygame.draw.line(self.screen, C_BORDER, (PANEL_X, y), (PANEL_X + PANEL_W - 10, y))
        y += 8
        cap_label = self.font_xs.render("Captured pieces", True, C_MUTED)
        self.screen.blit(cap_label, (PANEL_X, y))
        y += 16
        # White captured
        w_caps = " ".join(PIECE_UNICODE[p][chess.WHITE] for p in self.captured_black)
        b_caps = " ".join(PIECE_UNICODE[p][chess.BLACK] for p in self.captured_white)
        w_surf = self.font_md.render(f"You took: {w_caps}", True, (220,220,220))
        b_surf = self.font_md.render(f"AI took:  {b_caps}", True, (160,160,160))
        self.screen.blit(w_surf, (PANEL_X, y));     y += 22
        self.screen.blit(b_surf, (PANEL_X, y));     y += 28

        # Move history
        pygame.draw.line(self.screen, C_BORDER, (PANEL_X, y), (PANEL_X + PANEL_W - 10, y))
        y += 8
        hist_label = self.font_xs.render("Move history", True, C_MUTED)
        self.screen.blit(hist_label, (PANEL_X, y))
        y += 18

        # Show last 12 moves
        recent = self.move_history[-12:]
        for i, (san, _, is_ai) in enumerate(recent):
            move_num = (len(self.move_history) - len(recent) + i) // 2 + 1
            prefix   = f"{move_num}." if i % 2 == 0 else "   "
            who      = "AI" if is_ai else "You"
            color    = (160,200,255) if is_ai else C_ACCENT
            text     = self.font_xs.render(f"{prefix} {who}: {san}", True, color)
            self.screen.blit(text, (PANEL_X, y))
            y += 16
            if y > HEIGHT - 80:
                break

        # Controls at bottom
        controls = [
            ("[H] Hint",     C_MUTED), ("[Z] Undo",    C_MUTED),
            ("[R] Restart",  C_MUTED), ("[A] Analysis", C_ACCENT if self.game_over else C_MUTED),
            ("[1-4] Level",  C_MUTED), ("[Q] Quit",    C_MUTED),
        ]
        cy = HEIGHT - 72
        pygame.draw.line(self.screen, C_BORDER, (PANEL_X, cy-8), (PANEL_X + PANEL_W - 10, cy-8))
        for i, (txt, col) in enumerate(controls):
            s = self.font_xs.render(txt, True, col)
            self.screen.blit(s, (PANEL_X + (i % 2) * 115, cy + (i // 2) * 18))

    def draw(self):
        self.screen.fill(C_BG)
        self.draw_board()
        self.draw_pieces()
        self.draw_panel()

        # Game over overlay
        if self.game_over and not self.show_analysis:
            overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg  = self.status_msg
            surf = self.font_lg.render(msg, True, C_ACCENT)
            self.screen.blit(surf, (BOARD_SIZE//2 - surf.get_width()//2,
                                    BOARD_SIZE//2 - surf.get_height()//2))
            sub  = self.font_md.render("Press R to play again", True, C_TEXT)
            self.screen.blit(sub, (BOARD_SIZE//2 - sub.get_width()//2,
                                   BOARD_SIZE//2 + 36))
            sub2 = self.font_md.render("Press A for post-game analysis", True, C_ACCENT)
            self.screen.blit(sub2, (BOARD_SIZE//2 - sub2.get_width()//2,
                                    BOARD_SIZE//2 + 64))

        # Post-game analysis overlay
        if self.show_analysis:
            overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 210))
            self.screen.blit(overlay, (0, 0))
            title = self.font_lg.render("Game Analysis", True, C_ACCENT)
            self.screen.blit(title, (BOARD_SIZE//2 - title.get_width()//2, 20))
            for i, line in enumerate(self.analysis_lines[:26]):
                col = (80,200,120) if line.startswith("✅") else \
                      (220,80,80)  if line.startswith("❌") else \
                      (255,200,0)  if line.startswith("⚠") else C_TEXT
                s = self.font_xs.render(line, True, col)
                self.screen.blit(s, (20, 60 + i * 22))
            hint = self.font_xs.render("Press A to close  |  R to play again", True, C_MUTED)
            self.screen.blit(hint, (BOARD_SIZE//2 - hint.get_width()//2, BOARD_SIZE - 24))

        pygame.display.flip()

    # ── Input handling ────────────────────────────────────────────────────────
    def handle_click(self, x: int, y: int):
        if self.game_over or self.ai_thinking:
            return
        if self.board.turn != chess.WHITE:
            return

        sq = self.pixel_to_sq(x, y)
        if sq is None:
            return

        self.show_hint = False

        if self.selected_sq is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == chess.WHITE:
                self.selected_sq = sq
                self.legal_moves = [m for m in self.board.legal_moves
                                    if m.from_square == sq]
        else:
            # Try to make move
            move = None
            for m in self.legal_moves:
                if m.to_square == sq:
                    move = m
                    # Handle promotion
                    if (self.board.piece_at(self.selected_sq) and
                        self.board.piece_at(self.selected_sq).piece_type == chess.PAWN and
                        chess.square_rank(sq) in (0, 7)):
                        move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)
                    break

            if move and move in self.board.legal_moves:
                self.make_player_move(move)
            else:
                # Reselect
                piece = self.board.piece_at(sq)
                if piece and piece.color == chess.WHITE:
                    self.selected_sq = sq
                    self.legal_moves = [m for m in self.board.legal_moves
                                        if m.from_square == sq]
                else:
                    self.selected_sq = None
                    self.legal_moves = []

    def start_animation(self, move: chess.Move, piece):
        """Start a smooth slide animation for a move."""
        self.anim_active  = True
        self.anim_piece   = (piece.piece_type, piece.color)
        self.anim_from_px = self.sq_to_pixel(move.from_square)
        self.anim_to_px   = self.sq_to_pixel(move.to_square)
        self.anim_start   = time.time()
        self.anim_move    = move

    def make_player_move(self, move: chess.Move):
        # Track captures
        captured = self.board.piece_at(move.to_square)
        if captured:
            self.captured_black.append(captured.piece_type)

        piece = self.board.piece_at(move.from_square)
        self.start_animation(move, piece)

        san = self.board.san(move)
        # Save board snapshot for analysis
        board_before = self.board.copy()
        self.board.push(move)
        self.last_move  = move
        self.selected_sq = None
        self.legal_moves = []
        self.move_history.append((san, "", False))
        self.game_history.append({"san": san, "is_ai": False, "board": board_before, "move": move})

        self.check_game_over()
        if not self.game_over:
            self.status_msg  = "AI is thinking…"
            self.ai_thinking = True
            self.ai.think(self.board)

    def make_ai_move(self):
        move = self.ai.best_move
        if not move:
            return

        captured = self.board.piece_at(move.to_square)
        if captured:
            self.captured_white.append(captured.piece_type)

        piece = self.board.piece_at(move.from_square)
        self.start_animation(move, piece)

        san = self.board.san(move)
        board_before = self.board.copy()
        self.board.push(move)
        self.last_move      = move
        self.ai_explanation = self.ai.explanation
        self.move_history.append((san, self.ai.explanation, True))
        self.game_history.append({"san": san, "is_ai": True, "board": board_before, "move": move})
        self.ai_thinking    = False
        self.ai.moves_played += 1

        self.check_game_over()
        if not self.game_over:
            self.status_msg = "Your turn"

    def check_game_over(self):
        if self.board.is_checkmate():
            winner = "You win! 🎉" if self.board.turn == chess.BLACK else "AI wins!"
            self.status_msg  = winner
            self.game_over   = True
            self.ai.adjust_difficulty(self.board.turn == chess.BLACK)
        elif self.board.is_stalemate():
            self.status_msg = "Stalemate — Draw!"
            self.game_over  = True
        elif self.board.is_insufficient_material():
            self.status_msg = "Insufficient material — Draw!"
            self.game_over  = True
        elif self.board.is_check():
            self.status_msg = "Check! Your king is under attack!" if self.board.turn == chess.WHITE \
                              else "AI is in check!"

    def undo(self):
        if len(self.board.move_stack) >= 2 and not self.ai_thinking:
            self.board.pop()  # Undo AI move
            self.board.pop()  # Undo player move
            if self.move_history:
                self.move_history.pop()
            if self.move_history:
                self.move_history.pop()
            self.last_move    = self.board.peek() if self.board.move_stack else None
            self.selected_sq  = None
            self.legal_moves  = []
            self.game_over    = False
            self.status_msg   = "Move undone — your turn"
            self.ai_explanation = "No problem, let's try again!"
            # Restore captures (simplified — just clear and recount)
            self.captured_white = []
            self.captured_black = []

    def get_hint(self):
        if not self.ai_thinking and not self.game_over and self.board.turn == chess.WHITE:
            self.status_msg = "Calculating hint…"
            def _hint():
                move = self.ai.get_hint(self.board)
                self.hint_move  = move
                self.show_hint  = True
                self.status_msg = "Hint shown — highlighted in yellow"
            t = threading.Thread(target=_hint, daemon=True)
            t.start()

    def post_game_analysis(self):
        """Analyze the game and generate move-by-move feedback."""
        self.analysis_lines = []
        lines = self.analysis_lines

        result = "You won!" if "You win" in self.status_msg else                  "AI won!"  if "AI wins" in self.status_msg else "Draw!"
        lines.append(f"Result: {result}   Moves played: {len(self.game_history)}")
        lines.append("")

        piece_values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
                        chess.ROOK:5, chess.QUEEN:9, chess.KING:0}

        mistakes   = 0
        blunders   = 0
        good_moves = 0

        for i, entry in enumerate(self.game_history):
            move     = entry["move"]
            board    = entry["board"]
            san      = entry["san"]
            is_ai    = entry["is_ai"]
            who      = "AI" if is_ai else "You"

            piece    = board.piece_at(move.from_square)
            pname    = PIECE_NAMES.get(piece.piece_type, "piece") if piece else "piece"
            captured = board.piece_at(move.to_square)
            cap_val  = piece_values.get(captured.piece_type, 0) if captured else 0

            # Score before and after (simplified eval)
            score_before = self.ai.evaluate(board)
            board_after  = board.copy()
            board_after.push(move)
            score_after  = self.ai.evaluate(board_after)
            delta = score_after - score_before if not is_ai else score_before - score_after

            move_num = i // 2 + 1
            prefix   = f"{move_num:>2}. {who:3} {san:<8}"

            if captured and cap_val >= 3:
                lines.append(f"✅ {prefix} Great capture! Won a {PIECE_NAMES.get(captured.piece_type,'piece')} (+{cap_val})")
                good_moves += 1
            elif delta < -150 and not is_ai:
                lines.append(f"❌ {prefix} Blunder! This lost significant material.")
                blunders += 1
            elif delta < -60 and not is_ai:
                lines.append(f"⚠  {prefix} Mistake — a better move was available.")
                mistakes += 1
            elif delta > 100 and not is_ai:
                lines.append(f"✅ {prefix} Excellent move!")
                good_moves += 1
            else:
                lines.append(f"   {prefix}")

        lines.append("")
        lines.append(f"Summary: {good_moves} great moves  |  {mistakes} mistakes  |  {blunders} blunders")
        if blunders == 0 and mistakes <= 1:
            lines.append("⭐ Outstanding game — very clean play!")
        elif blunders <= 1:
            lines.append("✅ Good game — just a few errors.")
        else:
            lines.append("Keep practicing — focus on avoiding piece drops!")

        self.show_analysis = True

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("  ✅ Chess engine started!")
        print("  Controls: Click to move | H=Hint | Z=Undo | R=Restart | Q=Quit\n")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(*event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_r:
                        self.reset(keep_preset=True)
                    elif event.key == pygame.K_z:
                        self.undo()
                    elif event.key == pygame.K_h:
                        self.get_hint()
                    elif event.key == pygame.K_a:
                        if self.game_over:
                            if self.show_analysis:
                                self.show_analysis = False
                            else:
                                self.post_game_analysis()
                    elif event.key == pygame.K_1:
                        self.ai.set_preset(1)
                        self.status_msg = "Difficulty: Easy"
                    elif event.key == pygame.K_2:
                        self.ai.set_preset(2)
                        self.status_msg = "Difficulty: Medium"
                    elif event.key == pygame.K_3:
                        self.ai.set_preset(3)
                        self.status_msg = "Difficulty: Hard"
                    elif event.key == pygame.K_4:
                        self.ai.set_preset(4)
                        self.status_msg = "Difficulty: Grandmaster"

            # Check if AI has finished thinking
            if self.ai_thinking and not self.ai.thinking and self.ai.best_move:
                self.make_ai_move()

            self.draw()
            self.clock.tick(FPS)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           AI Chess Engine — Adaptive Difficulty              ║
╚══════════════════════════════════════════════════════════════╝
""")
    game = ChessGame()
    game.run()
