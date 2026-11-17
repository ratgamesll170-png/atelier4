// Atelier 4 — C++17 UCI chess engine (single file, no dependencies)
// Native build:   g++ -O3 -std=c++17 -pthread atelier4.cpp -o atelier4
// WebAssembly:    see build_wasm.sh (Emscripten)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <atomic>
#ifndef __EMSCRIPTEN__
#include <thread>
#endif

typedef uint64_t U64;
typedef U64 Bitboard;
typedef int Move;

enum Color { WHITE, BLACK };
enum PT { NO_PT = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

static const int MAX_PLY = 128;
static const int MATE = 32000;
static const int MATE_IN_MAX = MATE - MAX_PLY;
static const int INF = 32500;
static const int NONE_SC = 32767;

static const Bitboard FILE_A = 0x0101010101010101ULL, FILE_H = FILE_A << 7;
static const Bitboard RANK_1 = 0xFFULL, RANK_8 = RANK_1 << 56;

#define BIT(s) (1ULL << (s))
static inline int popcnt(Bitboard b) { return __builtin_popcountll(b); }
static inline int lsb(Bitboard b) { return __builtin_ctzll(b); }
static inline int poplsb(Bitboard &b) { int s = lsb(b); b &= b - 1; return s; }
static inline int fileOf(int s) { return s & 7; }
static inline int rankOf(int s) { return s >> 3; }
static inline int mk(int c, int pt) { return (c << 3) | pt; }
static inline int ptOf(int pc) { return pc & 7; }
static inline int colOf(int pc) { return pc >> 3; }

// ---------- moves: from(6) | to(6)<<6 | flag(4)<<12 ----------
enum { F_QUIET = 0, F_DOUBLE = 1, F_OO = 2, F_OOO = 3, F_CAP = 4, F_EP = 5, F_PROMO = 8 };
static inline int mfrom(Move m) { return m & 63; }
static inline int mto(Move m) { return (m >> 6) & 63; }
static inline int mflag(Move m) { return m >> 12; }
static inline Move mkmove(int f, int t, int fl) { return f | (t << 6) | (fl << 12); }
static inline bool isCap(Move m) { return (mflag(m) & 4) != 0; }
static inline bool isPromo(Move m) { return (mflag(m) & 8) != 0; }
static inline bool isTactical(Move m) { return mflag(m) >= 4; }
static inline int promoPT(Move m) { return (mflag(m) & 3) + 2; }

static std::string sqStr(int s) { std::string r; r += char('a' + fileOf(s)); r += char('1' + rankOf(s)); return r; }
static std::string moveStr(Move m) {
  if (!m) return "0000";
  std::string r = sqStr(mfrom(m)) + sqStr(mto(m));
  if (isPromo(m)) r += "nbrq"[promoPT(m) - 2];
  return r;
}

// ---------- random ----------
struct PRNG {
  U64 s;
  explicit PRNG(U64 seed) : s(seed) {}
  U64 next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; }
  U64 sparse() { return next() & next() & next(); }
};

// ---------- attack tables ----------
static Bitboard KnightAtt[64], KingAtt[64], PawnAtt[2][64];
static Bitboard BetweenBB[64][64], LineBB[64][64];
static Bitboard PassedMask[2][64], AdjFiles[8], FrontSpan[2][64];

struct Magic { Bitboard mask, magic; Bitboard *att; unsigned shift; };
static Magic RM[64], BM[64];
static Bitboard RT[0x19000], BT[0x1480];

static Bitboard slidingAtt(int sq, Bitboard occ, bool rook) {
  static const int rd[4][2] = {{1,0},{-1,0},{0,1},{0,-1}}, bd[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
  Bitboard a = 0;
  for (int d = 0; d < 4; d++) {
    int r = rankOf(sq), f = fileOf(sq);
    const int *dd = rook ? rd[d] : bd[d];
    for (;;) {
      r += dd[0]; f += dd[1];
      if (r < 0 || r > 7 || f < 0 || f > 7) break;
      int s = r * 8 + f; a |= BIT(s);
      if (occ & BIT(s)) break;
    }
  }
  return a;
}
static void initMagics(Magic *M, Bitboard *table, bool rook) {
  static Bitboard occ[4096], ref[4096]; static int epoch[4096]; int cnt = 0; size_t offset = 0;
  memset(epoch, 0, sizeof(epoch));
  PRNG rng(rook ? 728ULL : 8977ULL);
  for (int sq = 0; sq < 64; sq++) {
    Bitboard edges = ((RANK_1 | RANK_8) & ~(RANK_1 << (8 * rankOf(sq)))) | ((FILE_A | FILE_H) & ~(FILE_A << fileOf(sq)));
    Magic &m = M[sq];
    m.mask = slidingAtt(sq, 0, rook) & ~edges;
    m.shift = 64 - popcnt(m.mask);
    m.att = table + offset;
    int size = 0; Bitboard b = 0;
    do { occ[size] = b; ref[size] = slidingAtt(sq, b, rook); size++; b = (b - m.mask) & m.mask; } while (b);
    int i;
    do {
      do m.magic = rng.sparse(); while (popcnt((m.mask * m.magic) >> 56) < 6);
      ++cnt;
      for (i = 0; i < size; i++) {
        unsigned idx = unsigned((occ[i] * m.magic) >> m.shift);
        if (epoch[idx] < cnt) { epoch[idx] = cnt; m.att[idx] = ref[i]; }
        else if (m.att[idx] != ref[i]) break;
      }
    } while (i < size);
    offset += size;
  }
}
static inline Bitboard rookAtt(int s, Bitboard occ) { const Magic &m = RM[s]; return m.att[unsigned(((occ & m.mask) * m.magic) >> m.shift)]; }
static inline Bitboard bishopAtt(int s, Bitboard occ) { const Magic &m = BM[s]; return m.att[unsigned(((occ & m.mask) * m.magic) >> m.shift)]; }
static inline Bitboard queenAtt(int s, Bitboard occ) { return rookAtt(s, occ) | bishopAtt(s, occ); }

static inline Bitboard pawnAttsBB(Bitboard b, int c) {
  return c == WHITE ? (((b << 7) & ~FILE_H) | ((b << 9) & ~FILE_A)) : (((b >> 7) & ~FILE_A) | ((b >> 9) & ~FILE_H));
}

// ---------- zobrist ----------
static U64 ZP[16][64], ZC[16], ZE[8], ZS;
static int castleMask[64];

static void initTables() {
  for (int s = 0; s < 64; s++) {
    int r = rankOf(s), f = fileOf(s);
    KnightAtt[s] = KingAtt[s] = PawnAtt[0][s] = PawnAtt[1][s] = 0;
    static const int kn[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
    for (auto &d : kn) { int nr = r + d[0], nf = f + d[1]; if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) KnightAtt[s] |= BIT(nr * 8 + nf); }
    for (int dr = -1; dr <= 1; dr++) for (int df = -1; df <= 1; df++) {
      if (!dr && !df) continue; int nr = r + dr, nf = f + df;
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) KingAtt[s] |= BIT(nr * 8 + nf);
    }
    for (int df = -1; df <= 1; df += 2) {
      if (f + df < 0 || f + df > 7) continue;
      if (r + 1 < 8) PawnAtt[WHITE][s] |= BIT((r + 1) * 8 + f + df);
      if (r - 1 >= 0) PawnAtt[BLACK][s] |= BIT((r - 1) * 8 + f + df);
    }
  }
  initMagics(RM, RT, true);
  initMagics(BM, BT, false);
  for (int a = 0; a < 64; a++) for (int b = 0; b < 64; b++) {
    BetweenBB[a][b] = LineBB[a][b] = 0;
    if (a == b) continue;
    for (int rook = 0; rook < 2; rook++) {
      Bitboard aa = slidingAtt(a, 0, rook);
      if (aa & BIT(b)) {
        Bitboard bb2 = slidingAtt(b, 0, rook);
        LineBB[a][b] = (aa & bb2) | BIT(a) | BIT(b);
        BetweenBB[a][b] = slidingAtt(a, BIT(b), rook) & slidingAtt(b, BIT(a), rook);
        // line extends to board edge (full line)
        LineBB[a][b] = (slidingAtt(a, 0, rook) & slidingAtt(b, 0, rook)) | BIT(a) | BIT(b);
      }
    }
  }
  for (int f = 0; f < 8; f++) AdjFiles[f] = (f > 0 ? FILE_A << (f - 1) : 0) | (f < 7 ? FILE_A << (f + 1) : 0);
  for (int s = 0; s < 64; s++) {
    int f = fileOf(s), r = rankOf(s);
    FrontSpan[0][s] = FrontSpan[1][s] = 0;
    for (int rr = r + 1; rr < 8; rr++) FrontSpan[0][s] |= BIT(rr * 8 + f);
    for (int rr = r - 1; rr >= 0; rr--) FrontSpan[1][s] |= BIT(rr * 8 + f);
    PassedMask[0][s] = PassedMask[1][s] = 0;
    for (int rr = r + 1; rr < 8; rr++) for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ff++) PassedMask[0][s] |= BIT(rr * 8 + ff);
    for (int rr = r - 1; rr >= 0; rr--) for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ff++) PassedMask[1][s] |= BIT(rr * 8 + ff);
  }
  PRNG rng(0x9E3779B97F4A7C15ULL);
  for (int p = 0; p < 16; p++) for (int s = 0; s < 64; s++) ZP[p][s] = rng.next();
  for (int i = 0; i < 16; i++) ZC[i] = rng.next();
  for (int i = 0; i < 8; i++) ZE[i] = rng.next();
  ZS = rng.next();
  for (int s = 0; s < 64; s++) castleMask[s] = 15;
  castleMask[4] = 12; castleMask[7] = 14; castleMask[0] = 13;   // white: e1,h1,a1
  castleMask[60] = 3; castleMask[63] = 11; castleMask[56] = 7;  // black: e8,h8,a8
}

// ---------- position ----------
struct StateInfo { U64 key; int castle, ep, rule50, captured; Move move; };

static const int SEEV[7] = {0, 100, 320, 330, 500, 900, 0};

struct Position {
  int board[64];
  Bitboard bb[7], col[2];
  int stm, castle, ep, rule50, gamePly;
  U64 key;
  std::vector<StateInfo> hist;

  Bitboard occ() const { return col[0] | col[1]; }
  Bitboard pieces(int c, int pt) const { return bb[pt] & col[c]; }
  int kingSq(int c) const { return lsb(bb[KING] & col[c]); }

  void clear() {
    memset(board, 0, sizeof(board)); memset(bb, 0, sizeof(bb)); col[0] = col[1] = 0;
    stm = WHITE; castle = 0; ep = -1; rule50 = 0; gamePly = 0; key = 0; hist.clear();
  }
  void put(int s, int pc) { board[s] = pc; bb[ptOf(pc)] |= BIT(s); col[colOf(pc)] |= BIT(s); key ^= ZP[pc][s]; }
  void removeP(int s) { int pc = board[s]; board[s] = 0; bb[ptOf(pc)] &= ~BIT(s); col[colOf(pc)] &= ~BIT(s); key ^= ZP[pc][s]; }
  void movePiece(int f, int t) {
    int pc = board[f]; board[f] = 0; board[t] = pc;
    Bitboard m = BIT(f) | BIT(t); bb[ptOf(pc)] ^= m; col[colOf(pc)] ^= m; key ^= ZP[pc][f] ^ ZP[pc][t];
  }

  Bitboard attackersTo(int sq, Bitboard o) const {
    return (PawnAtt[BLACK][sq] & bb[PAWN] & col[WHITE]) | (PawnAtt[WHITE][sq] & bb[PAWN] & col[BLACK]) |
           (KnightAtt[sq] & bb[KNIGHT]) | (KingAtt[sq] & bb[KING]) |
           (bishopAtt(sq, o) & (bb[BISHOP] | bb[QUEEN])) | (rookAtt(sq, o) & (bb[ROOK] | bb[QUEEN]));
  }
  bool attackedBy(int sq, int by, Bitboard o) const { return (attackersTo(sq, o) & col[by]) != 0; }
  bool inCheck() const { return attackedBy(kingSq(stm), stm ^ 1, occ()); }
  Bitboard checkers() const { return attackersTo(kingSq(stm), occ()) & col[stm ^ 1]; }

  bool setFen(const std::string &fen) {
    clear();
    std::istringstream is(fen); std::string bs, side, cs, eps; int hm = 0, fm = 1;
    if (!(is >> bs >> side)) return false;
    is >> cs >> eps >> hm >> fm;
    int r = 7, f = 0;
    for (char ch : bs) {
      if (ch == '/') { r--; f = 0; if (r < 0) return false; }
      else if (ch >= '1' && ch <= '8') f += ch - '0';
      else {
        const char *pcs = "PNBRQK"; const char *p = strchr(pcs, toupper(ch));
        if (!p || !*p || f > 7) return false;
        int c = isupper((unsigned char)ch) ? WHITE : BLACK;
        put(r * 8 + f, mk(c, int(p - pcs) + 1)); f++;
      }
    }
    if (popcnt(bb[KING] & col[WHITE]) != 1 || popcnt(bb[KING] & col[BLACK]) != 1) return false;
    if (bb[PAWN] & (RANK_1 | RANK_8)) return false;
    stm = (side == "b") ? BLACK : WHITE;
    if (stm == BLACK) key ^= ZS;
    castle = 0;
    if (cs.find('K') != std::string::npos && board[4] == mk(WHITE, KING) && board[7] == mk(WHITE, ROOK)) castle |= 1;
    if (cs.find('Q') != std::string::npos && board[4] == mk(WHITE, KING) && board[0] == mk(WHITE, ROOK)) castle |= 2;
    if (cs.find('k') != std::string::npos && board[60] == mk(BLACK, KING) && board[63] == mk(BLACK, ROOK)) castle |= 4;
    if (cs.find('q') != std::string::npos && board[60] == mk(BLACK, KING) && board[56] == mk(BLACK, ROOK)) castle |= 8;
    key ^= ZC[castle];
    ep = -1;
    if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h' && (eps[1] == '3' || eps[1] == '6')) {
      int s = (eps[1] - '1') * 8 + (eps[0] - 'a');
      // only keep if the side to move can actually capture there
      if (PawnAtt[stm ^ 1][s] & pieces(stm, PAWN)) { ep = s; key ^= ZE[fileOf(s)]; }
    }
    rule50 = std::max(0, hm); gamePly = std::max(0, (fm - 1) * 2 + (stm == BLACK));
    // the side NOT to move must not be in check
    if (attackedBy(kingSq(stm ^ 1), stm, occ())) return false;
    return true;
  }

  std::string fen() const {
    std::string s;
    for (int r = 7; r >= 0; r--) {
      int e = 0;
      for (int f = 0; f < 8; f++) {
        int pc = board[r * 8 + f];
        if (!pc) e++;
        else { if (e) { s += char('0' + e); e = 0; } char c = "?PNBRQK"[ptOf(pc)]; s += colOf(pc) ? char(tolower(c)) : c; }
      }
      if (e) s += char('0' + e);
      if (r) s += '/';
    }
    s += stm ? " b " : " w ";
    std::string c; if (castle & 1) c += 'K'; if (castle & 2) c += 'Q'; if (castle & 4) c += 'k'; if (castle & 8) c += 'q';
    s += c.empty() ? "-" : c; s += ' ';
    s += ep >= 0 ? sqStr(ep) : "-";
    s += ' ' + std::to_string(rule50) + ' ' + std::to_string(gamePly / 2 + 1);
    return s;
  }

  // full legality test of a pseudo-legal move (castling is verified at generation)
  bool legal(Move m) const {
    int f = mfrom(m), t = mto(m), us = stm, them = us ^ 1, fl = mflag(m);
    if (fl == F_OO || fl == F_OOO) return true;
    int pc = board[f];
    Bitboard o = (occ() ^ BIT(f)) | BIT(t), capBB = 0;
    if (fl == F_EP) { int cs = t + (us == WHITE ? -8 : 8); o ^= BIT(cs); capBB = BIT(cs); }
    else if (fl & 4) capBB = BIT(t);
    int ks = ptOf(pc) == KING ? t : kingSq(us);
    // attackers of ks excluding captured piece, using modified occupancy
    Bitboard theirs = col[them] & ~capBB;
    Bitboard a = (PawnAtt[us][ks] & bb[PAWN] & theirs) | (KnightAtt[ks] & bb[KNIGHT] & theirs) | (KingAtt[ks] & bb[KING] & theirs) |
                 (bishopAtt(ks, o) & (bb[BISHOP] | bb[QUEEN]) & theirs) | (rookAtt(ks, o) & (bb[ROOK] | bb[QUEEN]) & theirs);
    return a == 0;
  }

  bool givesCheck(Move m) const {
    int f = mfrom(m), t = mto(m), us = stm, fl = mflag(m);
    int eks = kingSq(us ^ 1);
    if (fl == F_OO || fl == F_OOO || fl == F_EP) {
      Position q = *this; q.hist.clear(); q.make(m); return q.inCheck();
    }
    int pt = isPromo(m) ? promoPT(m) : ptOf(board[f]);
    Bitboard o = (occ() ^ BIT(f)) | BIT(t);
    Bitboard direct = 0;
    switch (pt) {
      case PAWN: direct = PawnAtt[us][t]; break;
      case KNIGHT: direct = KnightAtt[t]; break;
      case BISHOP: direct = bishopAtt(t, o); break;
      case ROOK: direct = rookAtt(t, o); break;
      case QUEEN: direct = queenAtt(t, o); break;
      default: break;
    }
    if (direct & BIT(eks)) return true;
    // discovered check
    Bitboard mine = col[us] & ~BIT(f);
    Bitboard sl = ((bishopAtt(eks, o) & (bb[BISHOP] | bb[QUEEN])) | (rookAtt(eks, o) & (bb[ROOK] | bb[QUEEN]))) & mine & ~BIT(t);
    return sl != 0;
  }

  void make(Move m) {
    StateInfo si{key, castle, ep, rule50, 0, m};
    int f = mfrom(m), t = mto(m), fl = mflag(m), us = stm, pc = board[f];
    if (ep >= 0) key ^= ZE[fileOf(ep)];
    ep = -1; rule50++;
    if (fl == F_EP) {
      int cs = t + (us == WHITE ? -8 : 8); si.captured = board[cs]; removeP(cs); rule50 = 0;
    } else if (fl & 4) { si.captured = board[t]; removeP(t); rule50 = 0; }
    if (ptOf(pc) == PAWN) rule50 = 0;
    if (isPromo(m)) { removeP(f); put(t, mk(us, promoPT(m))); }
    else movePiece(f, t);
    if (fl == F_OO) movePiece(t + 1, t - 1);
    else if (fl == F_OOO) movePiece(t - 2, t + 1);
    if (fl == F_DOUBLE) {
      int eps = (f + t) / 2;
      if (PawnAtt[us][eps] & pieces(us ^ 1, PAWN)) { ep = eps; key ^= ZE[fileOf(eps)]; }
    }
    key ^= ZC[castle];
    castle &= castleMask[f] & castleMask[t];
    key ^= ZC[castle];
    stm ^= 1; key ^= ZS; gamePly++;
    hist.push_back(si);
  }
  void unmake() {
    StateInfo si = hist.back(); hist.pop_back();
    Move m = si.move; int f = mfrom(m), t = mto(m), fl = mflag(m);
    stm ^= 1; gamePly--; int us = stm;
    if (isPromo(m)) { removeP(t); put(f, mk(us, PAWN)); }
    else movePiece(t, f);
    if (fl == F_OO) movePiece(t - 1, t + 1);
    else if (fl == F_OOO) movePiece(t + 1, t - 2);
    if (fl == F_EP) put(t + (us == WHITE ? -8 : 8), si.captured);
    else if (fl & 4) put(t, si.captured);
    key = si.key; castle = si.castle; ep = si.ep; rule50 = si.rule50;
  }
  void makeNull() {
    hist.push_back(StateInfo{key, castle, ep, rule50, 0, 0});
    if (ep >= 0) key ^= ZE[fileOf(ep)];
    ep = -1; stm ^= 1; key ^= ZS; rule50++; gamePly++;
  }
  void unmakeNull() {
    StateInfo si = hist.back(); hist.pop_back();
    stm ^= 1; gamePly--; key = si.key; castle = si.castle; ep = si.ep; rule50 = si.rule50;
  }

  // ---------- move generation (pseudo-legal) ----------
  // mode 0: all moves, 1: tactical (captures, promotions, ep)
  int gen(Move *ml, int mode) const {
    int n = 0; int us = stm, them = us ^ 1;
    Bitboard all = occ(), ownBB = col[us], enemy = col[them], empty = ~all;
    Bitboard pw = pieces(us, PAWN);
    int up = us == WHITE ? 8 : -8;
    Bitboard promoRank = us == WHITE ? RANK_8 : RANK_1;
    Bitboard thirdRank = us == WHITE ? (RANK_1 << 16) : (RANK_1 << 40);
    // pushes
    Bitboard one = (us == WHITE ? pw << 8 : pw >> 8) & empty;
    Bitboard promoPush = one & promoRank, quietPush = one & ~promoRank;
    while (promoPush) { int t = poplsb(promoPush); int f = t - up;
      ml[n++] = mkmove(f, t, F_PROMO + 3); if (mode == 0) { ml[n++] = mkmove(f, t, F_PROMO + 2); ml[n++] = mkmove(f, t, F_PROMO + 1); ml[n++] = mkmove(f, t, F_PROMO); } }
    if (mode == 0) {
      Bitboard q = quietPush;
      while (q) { int t = poplsb(q); ml[n++] = mkmove(t - up, t, F_QUIET); }
      Bitboard two = (us == WHITE ? (one & thirdRank) << 8 : (one & thirdRank) >> 8) & empty;
      while (two) { int t = poplsb(two); ml[n++] = mkmove(t - 2 * up, t, F_DOUBLE); }
    }
    // captures
    for (int side = 0; side < 2; side++) {
      Bitboard cap;
      int d;
      if (us == WHITE) { cap = (side == 0 ? ((pw << 7) & ~FILE_H) : ((pw << 9) & ~FILE_A)) & enemy; d = side == 0 ? 7 : 9; }
      else { cap = (side == 0 ? ((pw >> 9) & ~FILE_H) : ((pw >> 7) & ~FILE_A)) & enemy; d = side == 0 ? -9 : -7; }
      Bitboard pc = cap & promoRank, nc = cap & ~promoRank;
      while (pc) { int t = poplsb(pc); int f = t - d;
        for (int k = 3; k >= 0; k--) ml[n++] = mkmove(f, t, F_PROMO + 4 + k); }
      while (nc) { int t = poplsb(nc); ml[n++] = mkmove(t - d, t, F_CAP); }
    }
    if (ep >= 0) {
      Bitboard a = PawnAtt[them][ep] & pw;
      while (a) { int f = poplsb(a); ml[n++] = mkmove(f, ep, F_EP); }
    }
    Bitboard tgt = mode == 0 ? ~ownBB : enemy;
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
      Bitboard pcs = pieces(us, pt);
      while (pcs) {
        int f = poplsb(pcs);
        Bitboard a = pt == KNIGHT ? KnightAtt[f] : pt == BISHOP ? bishopAtt(f, all) : pt == ROOK ? rookAtt(f, all) : queenAtt(f, all);
        a &= tgt;
        while (a) { int t = poplsb(a); ml[n++] = mkmove(f, t, (enemy & BIT(t)) ? F_CAP : F_QUIET); }
      }
    }
    { int f = kingSq(us); Bitboard a = KingAtt[f] & tgt;
      while (a) { int t = poplsb(a); ml[n++] = mkmove(f, t, (enemy & BIT(t)) ? F_CAP : F_QUIET); } }
    if (mode == 0) {
      if (us == WHITE) {
        if ((castle & 1) && !(all & (BIT(5) | BIT(6))) && !attackedBy(4, BLACK, all) && !attackedBy(5, BLACK, all) && !attackedBy(6, BLACK, all)) ml[n++] = mkmove(4, 6, F_OO);
        if ((castle & 2) && !(all & (BIT(1) | BIT(2) | BIT(3))) && !attackedBy(4, BLACK, all) && !attackedBy(3, BLACK, all) && !attackedBy(2, BLACK, all)) ml[n++] = mkmove(4, 2, F_OOO);
      } else {
        if ((castle & 4) && !(all & (BIT(61) | BIT(62))) && !attackedBy(60, WHITE, all) && !attackedBy(61, WHITE, all) && !attackedBy(62, WHITE, all)) ml[n++] = mkmove(60, 62, F_OO);
        if ((castle & 8) && !(all & (BIT(57) | BIT(58) | BIT(59))) && !attackedBy(60, WHITE, all) && !attackedBy(59, WHITE, all) && !attackedBy(58, WHITE, all)) ml[n++] = mkmove(60, 58, F_OOO);
      }
    }
    return n;
  }

  int genLegal(Move *ml) const {
    Move tmp[256]; int n = gen(tmp, 0), k = 0;
    for (int i = 0; i < n; i++) if (legal(tmp[i])) ml[k++] = tmp[i];
    return k;
  }

  bool seeGE(Move m, int th) const {
    int f = mfrom(m), t = mto(m);
    if (mflag(m) == F_OO || mflag(m) == F_OOO) return th <= 0;
    int cap = mflag(m) == F_EP ? PAWN : ptOf(board[t]);
    int swap = SEEV[cap] - th;
    if (isPromo(m)) swap += SEEV[promoPT(m)] - SEEV[PAWN];
    if (swap < 0) return false;
    int mover = isPromo(m) ? promoPT(m) : ptOf(board[f]);
    swap = SEEV[mover] - swap;
    if (swap <= 0) return true;
    Bitboard o = occ() ^ BIT(f) ^ BIT(t);
    if (mflag(m) == F_EP) o ^= BIT(t + (stm == WHITE ? -8 : 8));
    int s = stm; Bitboard attackers = attackersTo(t, o); int res = 1;
    for (;;) {
      s ^= 1; attackers &= o;
      Bitboard sa = attackers & col[s];
      if (!sa) break;
      res ^= 1;
      Bitboard b;
      if ((b = sa & bb[PAWN])) { if ((swap = SEEV[PAWN] - swap) < res) break; o ^= BIT(lsb(b)); attackers |= bishopAtt(t, o) & (bb[BISHOP] | bb[QUEEN]); }
      else if ((b = sa & bb[KNIGHT])) { if ((swap = SEEV[KNIGHT] - swap) < res) break; o ^= BIT(lsb(b)); }
      else if ((b = sa & bb[BISHOP])) { if ((swap = SEEV[BISHOP] - swap) < res) break; o ^= BIT(lsb(b)); attackers |= bishopAtt(t, o) & (bb[BISHOP] | bb[QUEEN]); }
      else if ((b = sa & bb[ROOK])) { if ((swap = SEEV[ROOK] - swap) < res) break; o ^= BIT(lsb(b)); attackers |= rookAtt(t, o) & (bb[ROOK] | bb[QUEEN]); }
      else if ((b = sa & bb[QUEEN])) { if ((swap = SEEV[QUEEN] - swap) < res) break; o ^= BIT(lsb(b)); attackers |= (bishopAtt(t, o) & (bb[BISHOP] | bb[QUEEN])) | (rookAtt(t, o) & (bb[ROOK] | bb[QUEEN])); }
      else return (attackers & ~col[s]) ? (res ^ 1) != 0 : res != 0;
    }
    return res != 0;
  }

  bool insufficient() const {
    if (bb[PAWN] | bb[ROOK] | bb[QUEEN]) return false;
    int minors = popcnt(bb[KNIGHT] | bb[BISHOP]);
    if (minors <= 1) return true;
    if (!bb[KNIGHT]) {                       // only bishops: all on one colour
      const Bitboard lightSq = 0x55AA55AA55AA55AAULL;
      return !(bb[BISHOP] & lightSq) || !(bb[BISHOP] & ~lightSq);
    }
    return false;
  }

  // repetition: ply = distance from search root
  bool repetition(int ply) const {
    int n = int(hist.size()), lim = std::min(rule50, n), cnt = 0;
    for (int k = 4; k <= lim; k += 2) {
      if (hist[n - k].key == key) {
        if (k <= ply) return true;
        if (++cnt >= 2) return true;
      }
    }
    return false;
  }
  int nonPawnMat(int c) const {
    return popcnt(pieces(c, KNIGHT)) * 320 + popcnt(pieces(c, BISHOP)) * 330 + popcnt(pieces(c, ROOK)) * 500 + popcnt(pieces(c, QUEEN)) * 900;
  }
  bool hasPieces(int c) const { return (col[c] & ~(bb[PAWN] | bb[KING])) != 0; }
  Move parseMove(const std::string &s) const {
    Move ml[256]; int n = genLegal(ml);
    for (int i = 0; i < n; i++) if (moveStr(ml[i]) == s) return ml[i];
    return 0;
  }
};

static U64 perft(Position &p, int d) {
  Move ml[256]; int n = p.gen(ml, 0); U64 c = 0;
  for (int i = 0; i < n; i++) {
    if (!p.legal(ml[i])) continue;
    if (d == 1) { c++; continue; }
    p.make(ml[i]); c += perft(p, d - 1); p.unmake();
  }
  return c;
}

// =====================================================================
//                              EVALUATION
// =====================================================================
static int PST_MG[7][64], PST_EG[7][64];
static const int VAL_MG[7] = {0, 82, 337, 365, 477, 1025, 0};
static const int VAL_EG[7] = {0, 94, 281, 297, 512, 936, 0};
static const int PHASE_W[7] = {0, 0, 1, 1, 2, 4, 0};
static const int PASS_MG[8] = {0, 2, 6, 14, 30, 56, 90, 0};
static const int PASS_EG[8] = {0, 5, 12, 28, 52, 90, 140, 0};
static const Bitboard LIGHT_SQ = 0x55AA55AA55AA55AAULL;

static void initEval() {
  static const int pMg[8] = {0, -6, -8, 0, 8, 20, 30, 0}, pEg[8] = {0, 0, 2, 8, 20, 38, 60, 0};
  static const int kFile[8] = {0, 14, 10, -12, -12, 2, 22, 6};
  for (int s = 0; s < 64; s++) {
    int f = fileOf(s), r = rankOf(s);
    int d = std::min(f, 7 - f) + std::min(r, 7 - r);
    int pm = pMg[r] + (((f == 3 || f == 4) && r >= 2 && r <= 4) ? 8 : ((f == 2 || f == 5) && r >= 2 && r <= 4) ? 3 : 0) + ((f == 0 || f == 7) ? -4 : 0);
    int pe = pEg[r] + ((f == 0 || f == 7) ? -3 : 0);
    if (r == 0 || r == 7) pm = pe = 0;
    PST_MG[PAWN][s] = pm; PST_EG[PAWN][s] = pe;
    PST_MG[KNIGHT][s] = d * 7 - 22; PST_EG[KNIGHT][s] = d * 6 - 18;
    PST_MG[BISHOP][s] = d * 3 - 8 - (r == 0 ? 4 : 0); PST_EG[BISHOP][s] = d * 3 - 8;
    PST_MG[ROOK][s] = (f == 3 || f == 4) ? 5 : (f == 2 || f == 5) ? 2 : 0; PST_EG[ROOK][s] = 0;
    PST_MG[QUEEN][s] = d * 2 - 6 - (r == 0 ? 3 : 0); PST_EG[QUEEN][s] = d * 3 - 9;
    PST_MG[KING][s] = r == 0 ? kFile[f] : -22 * r - 10; PST_EG[KING][s] = d * 9 - 27;
  }
}

static inline int cheb(int a, int b) { return std::max(std::abs(fileOf(a) - fileOf(b)), std::abs(rankOf(a) - rankOf(b))); }
static inline int manh(int a, int b) { return std::abs(fileOf(a) - fileOf(b)) + std::abs(rankOf(a) - rankOf(b)); }

static int evaluate(const Position &p) {
  int mg[2] = {0, 0}, eg[2] = {0, 0}, phase = 0, npm[2] = {0, 0};
  const Bitboard all = p.occ();
  Bitboard pawns[2] = {p.pieces(WHITE, PAWN), p.pieces(BLACK, PAWN)};
  Bitboard pAtt[2] = {pawnAttsBB(pawns[0], WHITE), pawnAttsBB(pawns[1], BLACK)};
  Bitboard att[2] = {pAtt[0], pAtt[1]};
  int ks[2] = {p.kingSq(WHITE), p.kingSq(BLACK)};
  Bitboard kzone[2];
  for (int c = 0; c < 2; c++) {
    Bitboard z = KingAtt[ks[c]] | BIT(ks[c]);
    z |= (c == WHITE) ? (z << 8) : (z >> 8);
    kzone[c] = z; att[c] |= KingAtt[ks[c]];
  }
  int atkCnt[2] = {0, 0}, atkUnits[2] = {0, 0};
  Bitboard weakNonPawn[2] = {0, 0};

  for (int c = 0; c < 2; c++) {
    const int them = c ^ 1, sgn = c == WHITE ? 0 : 56;
    const Bitboard mobArea = ~p.col[c] & ~pAtt[them];
    // king + pawn psq
    { int s = ks[c]; mg[c] += PST_MG[KING][s ^ sgn]; eg[c] += PST_EG[KING][s ^ sgn]; }
    Bitboard pb = pawns[c];
    while (pb) { int s = poplsb(pb); mg[c] += VAL_MG[PAWN] + PST_MG[PAWN][s ^ sgn]; eg[c] += VAL_EG[PAWN] + PST_EG[PAWN][s ^ sgn]; }
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
      Bitboard pcs = p.pieces(c, pt);
      while (pcs) {
        int s = poplsb(pcs), rel = rankOf(s ^ sgn), f = fileOf(s);
        mg[c] += VAL_MG[pt] + PST_MG[pt][s ^ sgn]; eg[c] += VAL_EG[pt] + PST_EG[pt][s ^ sgn];
        phase += PHASE_W[pt]; npm[c] += SEEV[pt];
        Bitboard a;
        if (pt == KNIGHT) a = KnightAtt[s];
        else if (pt == BISHOP) a = bishopAtt(s, all);
        else if (pt == ROOK) a = rookAtt(s, all & ~(p.pieces(c, ROOK) | p.pieces(c, QUEEN)));
        else a = queenAtt(s, all);
        att[c] |= a;
        int cnt = popcnt(a & mobArea);
        Bitboard z = a & kzone[them];
        if (z) { atkCnt[c]++; atkUnits[c] += (pt == KNIGHT || pt == BISHOP ? 2 : pt == ROOK ? 3 : 5) * popcnt(z); }
        if (pt == KNIGHT) {
          mg[c] += (cnt - 4) * 4; eg[c] += (cnt - 4) * 3;
          if (rel >= 3 && rel <= 5 && !(pawns[them] & PassedMask[c][s] & AdjFiles[f])) {
            bool sup = (pAtt[c] & BIT(s)) != 0;
            mg[c] += sup ? 22 : 9; eg[c] += sup ? 14 : 5;
          }
        } else if (pt == BISHOP) {
          mg[c] += (cnt - 6) * 4; eg[c] += (cnt - 6) * 4;
          if (rel >= 3 && rel <= 5 && !(pawns[them] & PassedMask[c][s] & AdjFiles[f])) {
            bool sup = (pAtt[c] & BIT(s)) != 0; mg[c] += sup ? 12 : 3; eg[c] += sup ? 8 : 2;
          }
          int same = popcnt(pawns[c] & ((LIGHT_SQ & BIT(s)) ? LIGHT_SQ : ~LIGHT_SQ));
          mg[c] -= same * 2; eg[c] -= same * 3;
        } else if (pt == ROOK) {
          mg[c] += (cnt - 7) * 2; eg[c] += (cnt - 7) * 4;
          Bitboard fb = FILE_A << f;
          if (!(fb & pawns[c])) { if (!(fb & pawns[them])) { mg[c] += 22; eg[c] += 10; } else { mg[c] += 11; eg[c] += 6; } }
          if (rel == 6 && (rankOf(ks[them] ^ sgn) == 7 || (pawns[them] & (RANK_1 << (8 * (c == WHITE ? 6 : 1)))))) { mg[c] += 15; eg[c] += 25; }
          if (cnt <= 3 && rel == 0 && rankOf(ks[c] ^ sgn) == 0 && !(p.castle & (c == WHITE ? 3 : 12))) {
            int kf = fileOf(ks[c]);
            if ((f > kf && kf >= 4) || (f < kf && kf <= 3)) { mg[c] -= (4 - cnt) * 14; eg[c] -= (4 - cnt) * 5; }
          }
        } else {
          mg[c] += (cnt - 14); eg[c] += (cnt - 14) * 2;
        }
      }
    }
    if (popcnt(p.pieces(c, BISHOP)) >= 2) { mg[c] += 28; eg[c] += 48; }
  }

  // threats, hanging pieces, king safety, pawns
  for (int c = 0; c < 2; c++) {
    const int them = c ^ 1, sgn = c == WHITE ? 0 : 56;
    Bitboard minors = p.pieces(c, KNIGHT) | p.pieces(c, BISHOP);
    mg[c] -= 26 * popcnt(minors & pAtt[them]); eg[c] -= 20 * popcnt(minors & pAtt[them]);
    mg[c] -= 38 * popcnt(p.pieces(c, ROOK) & pAtt[them]); eg[c] -= 28 * popcnt(p.pieces(c, ROOK) & pAtt[them]);
    mg[c] -= 48 * popcnt(p.pieces(c, QUEEN) & pAtt[them]); eg[c] -= 34 * popcnt(p.pieces(c, QUEEN) & pAtt[them]);
    Bitboard hang = (minors | p.pieces(c, ROOK)) & att[them] & ~att[c] & ~pAtt[them];
    mg[c] -= 14 * popcnt(hang); eg[c] -= 10 * popcnt(hang);
    Bitboard qh = p.pieces(c, QUEEN) & att[them] & ~att[c] & ~pAtt[them];
    mg[c] -= 20 * popcnt(qh); eg[c] -= 14 * popcnt(qh);

    // king danger from opponent attackers
    {
      int d = atkUnits[them];
      if (!p.pieces(them, QUEEN)) d = d * 2 / 3;
      if (atkCnt[them] < 2) d /= 2;
      int pen = std::min(d * d / 10, 380);
      mg[c] -= pen; eg[c] -= pen / 6;
    }
    // pawn shield / storm (middlegame only, king near its back rank)
    {
      int k = ks[c], kr = rankOf(k ^ sgn), kf = fileOf(k);
      if (kr <= 1) {
        Bitboard ahead = c == WHITE ? (~0ULL << (8 * (rankOf(k) + 1))) : ((1ULL << (8 * rankOf(k))) - 1);
        static const int shield[5] = {0, 14, 9, 3, 0};
        int sc = 0;
        for (int f = std::max(0, kf - 1); f <= std::min(7, kf + 1); f++) {
          Bitboard fb = (FILE_A << f) & ahead;
          Bitboard own = pawns[c] & fb;
          if (!own) { sc -= 20; if (!(pawns[them] & fb)) sc -= 8; }
          else {
            int pr = c == WHITE ? lsb(own) : 63 - __builtin_clzll(own);
            int dist = std::abs(rankOf(pr) - rankOf(k)); sc += shield[std::min(dist, 4)];
          }
          Bitboard en = pawns[them] & fb;
          if (en) {
            int pr = c == WHITE ? lsb(en) : 63 - __builtin_clzll(en);
            int dist = std::abs(rankOf(pr) - rankOf(k));
            if (dist == 2) sc -= 12; else if (dist == 3) sc -= 6; else if (dist == 4) sc -= 2;
          }
        }
        mg[c] += sc;
      }
    }
    // pawn structure
    Bitboard pb = pawns[c];
    while (pb) {
      int s = poplsb(pb), f = fileOf(s), r = rankOf(s ^ sgn);
      Bitboard fb = FILE_A << f;
      bool doubled = (pawns[c] & FrontSpan[c][s]) != 0;
      bool isolated = !(pawns[c] & AdjFiles[f]);
      bool supported = (pAtt[c] & BIT(s)) != 0;
      bool phalanx = (pawns[c] & AdjFiles[f] & (RANK_1 << (8 * rankOf(s)))) != 0;
      bool opposed = (pawns[them] & FrontSpan[c][s]) != 0;
      bool passed = !(pawns[them] & PassedMask[c][s]);
      int stop = s + (c == WHITE ? 8 : -8);
      if (doubled) { mg[c] -= 9; eg[c] -= 15; }
      if (isolated) { mg[c] -= 8 + (opposed ? 0 : 5); eg[c] -= 12; }
      else if (!supported && !phalanx) {
        // backward: no friendly pawn level or behind on adjacent files and stop square controlled by enemy pawn
        Bitboard behindAdj = pawns[c] & AdjFiles[f] & PassedMask[them][s];
        if (!behindAdj && (pAtt[them] & BIT(stop))) { mg[c] -= 8; eg[c] -= 10; }
      }
      if (supported || phalanx) {
        static const int base[8] = {0, 2, 4, 7, 14, 28, 50, 0};
        int b = base[r] * (1 + (supported ? 1 : 0) + (phalanx ? 1 : 0)) / 2;
        mg[c] += b; eg[c] += b * 3 / 4;
      }
      if (passed && !doubled) {
        int pm = PASS_MG[r], pe = PASS_EG[r];
        mg[c] += pm; eg[c] += pe;
        if (supported) { mg[c] += r * 2; eg[c] += r * 4; }
        if (all & BIT(stop)) {
          if (p.col[them] & BIT(stop)) { mg[c] -= pm / 4; eg[c] -= pe / 4; }
        } else if (r >= 2 && (att[them] & BIT(stop)) && !(att[c] & BIT(stop))) { mg[c] -= pm / 8; eg[c] -= pe / 8; }
        if (r >= 3) {
          eg[c] += (cheb(ks[them], stop) - cheb(ks[c], stop)) * (r - 1) * 2;
          Bitboard behind = FrontSpan[them][s] & all;
          if (behind) {
            int bq = c == WHITE ? 63 - __builtin_clzll(behind) : lsb(behind);
            int bp = p.board[bq];
            if (ptOf(bp) == ROOK) { if (colOf(bp) == c) { mg[c] += r * 2; eg[c] += r * 5; } else { mg[c] -= r; eg[c] -= r * 3; } }
          }
        }
        // rule of the square when the opponent has only king + pawns
        if (npm[them] == 0 && r >= 2 && !(FrontSpan[c][s] & all)) {
          int promo = (c == WHITE ? 56 : 0) + f;
          int dist = std::min(5, 7 - r);
          int kd = cheb(ks[them], promo) - (p.stm == them ? 1 : 0);
          if (kd > dist) eg[c] += 350 + r * 20;
        }
      }
    }
  }

  // endgame scaling and mop-up
  int egS = eg[0] - eg[1], mgS = mg[0] - mg[1];
  int strong = egS > 0 ? WHITE : BLACK, weak = strong ^ 1;
  int scale = 64;
  if (!pawns[strong]) {
    if (npm[strong] - npm[weak] < 400) scale = 8;
    if (npm[weak] == 0 && npm[strong] == 640 && !p.pieces(strong, BISHOP)) scale = 1;   // KNN vs K
  } else if (popcnt(pawns[strong]) == 1 && !pawns[weak] && npm[strong] == npm[weak] && npm[strong] <= 330) {
    scale = 40;
  }
  if (npm[0] == 330 && npm[1] == 330 && popcnt(p.bb[BISHOP]) == 2 &&
      (((p.pieces(WHITE, BISHOP) & LIGHT_SQ) != 0) != ((p.pieces(BLACK, BISHOP) & LIGHT_SQ) != 0)))
    scale = std::min(scale, std::abs(popcnt(pawns[0]) - popcnt(pawns[1])) <= 1 ? 36 : 50);
  if (!pawns[weak] && npm[strong] - npm[weak] >= 400 && npm[strong] > 0) {
    int bonus;
    if (npm[weak] == 0 && p.pieces(strong, BISHOP) && p.pieces(strong, KNIGHT) && popcnt(p.col[strong]) == 3) {
      // KBNK: drive the king to a corner of the bishop's colour
      bool lightB = (p.pieces(strong, BISHOP) & LIGHT_SQ) != 0;
      int c1 = lightB ? 7 : 0, c2 = lightB ? 56 : 63, wk = ks[weak];
      int cd = std::min(manh(wk, c1), manh(wk, c2));
      bonus = (14 - cd) * 18 + (7 - cheb(ks[0], ks[1])) * 8;
    } else {
      int wk = ks[weak];
      int cdist = (std::abs(2 * fileOf(wk) - 7) + std::abs(2 * rankOf(wk) - 7)) / 2;
      bonus = cdist * 14 + (7 - cheb(ks[0], ks[1])) * 8;
    }
    if (strong == WHITE) egS += bonus; else egS -= bonus;
  }
  egS = egS * scale / 64;
  phase = std::min(phase, 24);
  int score = (mgS * phase + egS * (24 - phase)) / 24;
  score += p.stm == WHITE ? 10 : -10;
  return p.stm == WHITE ? score : -score;
}

// =====================================================================
//                         TRANSPOSITION TABLE
// =====================================================================
enum { B_NONE = 0, B_UPPER = 1, B_LOWER = 2, B_EXACT = 3 };
struct TTEntry { U64 key; int16_t score, eval; uint16_t move; int8_t depth; uint8_t gb; };
struct TTable {
  TTEntry *t = nullptr; size_t nb = 0; uint8_t gen = 0;
  ~TTable() { free(t); }
  void resize(size_t mb) {
    free(t); t = nullptr;
    size_t bytes = mb * 1024 * 1024, n = 1;
    while (n * 4 * sizeof(TTEntry) * 2 <= bytes) n <<= 1;       // buckets of 4 entries, power of two
    nb = n; t = (TTEntry *)calloc(nb * 4, sizeof(TTEntry));
    if (!t) { nb = 1 << 10; t = (TTEntry *)calloc(nb * 4, sizeof(TTEntry)); }
    gen = 0;
  }
  void clear() { memset(t, 0, nb * 4 * sizeof(TTEntry)); gen = 0; }
  void newSearch() { gen = (gen + 1) & 63; }
  TTEntry *bucket(U64 k) const { return t + (k & (nb - 1)) * 4; }
  bool probe(U64 k, TTEntry &out) const {
    TTEntry *b = bucket(k);
    for (int i = 0; i < 4; i++) if (b[i].key == k && (b[i].gb & 3)) { out = b[i]; return true; }
    return false;
  }
  void store(U64 k, int depth, int bound, int score, int eval, Move m) {
    TTEntry *b = bucket(k), *r = b;
    for (int i = 0; i < 4; i++) {
      if (b[i].key == k || !(b[i].gb & 3)) { r = &b[i]; break; }
      int age = (gen - (b[i].gb >> 2)) & 63, ra = (gen - (r->gb >> 2)) & 63;
      if (b[i].depth - 8 * age < r->depth - 8 * ra) r = &b[i];
    }
    if (r->key == k) {
      if (!m) m = r->move;
      if (bound != B_EXACT && depth < r->depth - 3 && ((gen - (r->gb >> 2)) & 63) == 0) { if (!r->move) r->move = (uint16_t)m; return; }
    }
    r->key = k; r->score = (int16_t)score; r->eval = (int16_t)eval; r->move = (uint16_t)m;
    r->depth = (int8_t)std::max(-1, std::min(depth, 127)); r->gb = (uint8_t)((gen << 2) | bound);
  }
  int hashfull() const {
    int c = 0; size_t n = std::min<size_t>(nb, 250);
    for (size_t i = 0; i < n; i++) for (int j = 0; j < 4; j++) { const TTEntry &e = t[i * 4 + j]; if ((e.gb & 3) && (e.gb >> 2) == gen) c++; }
    return int(c * 1000 / (n * 4));
  }
};
static inline int scoreToTT(int s, int ply) { return s >= MATE_IN_MAX ? s + ply : s <= -MATE_IN_MAX ? s - ply : s; }
static inline int scoreFromTT(int s, int ply) { return s >= MATE_IN_MAX ? s - ply : s <= -MATE_IN_MAX ? s + ply : s; }

// =====================================================================
//                                SEARCH
// =====================================================================
static TTable TT;
static std::atomic<bool> g_stop(false);
static std::atomic<bool> g_searching(false);

struct Limits {
  long long wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1, nodes = 0;
  int depth = 0, movestogo = 0; bool infinite = false;
};

struct Stack {
  Move move; int eval; Move killers[2]; Move excluded; int16_t (*cont)[64]; bool inCheck;
};

static int OPT_MULTIPV = 1, OPT_OVERHEAD = 30;
static int LMR[64][64];

static std::chrono::steady_clock::time_point nowTP() { return std::chrono::steady_clock::now(); }
static long long msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(nowTP() - t).count();
}

static void say(const std::string &s) { fputs(s.c_str(), stdout); fputc('\n', stdout); fflush(stdout); }

struct Searcher {
  Position pos;
  Stack stackBuf[MAX_PLY + 12]; Stack *ss = stackBuf + 6;
  int16_t mainHist[2][64][64];
  int16_t capHist[16][64][7];
  int16_t contHist[16][64][16][64];
  Move counterMv[16][64];
  Move pv[MAX_PLY + 2][MAX_PLY + 2]; int pvLen[MAX_PLY + 2];
  U64 nodes = 0; int seldepth = 0;
  std::chrono::steady_clock::time_point t0;
  long long hardMs = 0, softMs = 0; U64 nodeLimit = 0; bool timed = false;
  int nmpMinPly = 0, nmpColor = 0;
  Move rootExcl[256]; int nExcl = 0;
  Move rootBest = 0; int rootScore = 0; bool rootHasBest = false;
  int multiPVidx = 0;

  void clearHistory() {
    memset(mainHist, 0, sizeof(mainHist)); memset(capHist, 0, sizeof(capHist));
    memset(contHist, 0, sizeof(contHist)); memset(counterMv, 0, sizeof(counterMv));
  }
  static void hupd(int16_t &v, int bonus) {
    int b = std::max(-16000, std::min(16000, bonus));
    v += b - v * std::abs(b) / 16384;
  }
  void checkTime() {
    if (g_stop.load(std::memory_order_relaxed)) return;
    if (nodeLimit && nodes >= nodeLimit) { g_stop = true; return; }
    if (timed && msSince(t0) >= hardMs) g_stop = true;
  }

  int quietScore(int ply, Move m) const {
    int pc = pos.board[mfrom(m)], t = mto(m);
    int s = mainHist[pos.stm][mfrom(m)][t];
    s += 2 * ss[ply - 1].cont[pc][t] + ss[ply - 2].cont[pc][t] + ss[ply - 4].cont[pc][t] / 2;
    return s;
  }

  int scoreMoves(const Move *ml, int *sc, int n, Move ttMove, int ply, bool evasion) {
    Move counter = 0;
    if (ply >= 1 && ss[ply - 1].move) { Move pm = ss[ply - 1].move; counter = counterMv[pos.board[mto(pm)]][mto(pm)]; }
    for (int i = 0; i < n; i++) {
      Move m = ml[i]; int f = mfrom(m), t = mto(m);
      if (m == ttMove) { sc[i] = 1 << 30; continue; }
      if (isTactical(m)) {
        int vic = mflag(m) == F_EP ? PAWN : ptOf(pos.board[t]);
        int att = ptOf(pos.board[f]);
        if (isPromo(m)) { sc[i] = promoPT(m) == QUEEN ? 2000000 + (vic ? SEEV[vic] : 0) : -300000 + promoPT(m); continue; }
        int ch = capHist[pos.board[f]][t][vic];
        bool good = SEEV[vic] >= SEEV[att] || pos.seeGE(m, 0);
        sc[i] = good ? 1000000 + 16 * SEEV[vic] - att + ch / 16 : -200000 + 16 * SEEV[vic] - att + ch / 16;
      } else if (m == ss[ply].killers[0]) sc[i] = 900000;
      else if (m == ss[ply].killers[1]) sc[i] = 800000;
      else if (m == counter) sc[i] = 700000;
      else sc[i] = quietScore(ply, m);
    }
    (void)evasion; return n;
  }
  static Move pickBest(Move *ml, int *sc, int n, int i) {
    int b = i;
    for (int j = i + 1; j < n; j++) if (sc[j] > sc[b]) b = j;
    std::swap(ml[i], ml[b]); std::swap(sc[i], sc[b]);
    return ml[i];
  }

  bool isDrawNode(int ply) {
    if (pos.rule50 >= 100) {
      if (!pos.inCheck()) return true;
      Move ml[256]; return pos.genLegal(ml) > 0;
    }
    if (pos.repetition(ply)) return true;
    return pos.insufficient();
  }

  int qsearch(int alpha, int beta, int ply, bool isPV) {
    pvLen[ply] = 0;
    if ((++nodes & 1023) == 0) checkTime();
    if (g_stop) return 0;
    if (ply > seldepth) seldepth = ply;
    if (ply >= MAX_PLY - 1) return evaluate(pos);
    if (ply > 0 && isDrawNode(ply)) return 0;
    bool inCheck = pos.inCheck();
    ss[ply].inCheck = inCheck;
    TTEntry te; bool ttHit = TT.probe(pos.key, te);
    Move ttMove = ttHit ? te.move : 0; int ttScore = NONE_SC;
    if (ttHit) {
      ttScore = scoreFromTT(te.score, ply);
      int bd = te.gb & 3;
      if (!isPV && ttScore != NONE_SC && ((bd == B_EXACT) || (bd == B_LOWER && ttScore >= beta) || (bd == B_UPPER && ttScore <= alpha))) return ttScore;
    }
    int best, standPat = NONE_SC;
    if (inCheck) best = -MATE + ply;
    else {
      standPat = ttHit && te.eval != NONE_SC ? te.eval : evaluate(pos);
      best = standPat;
      if (ttHit && ttScore != NONE_SC) { int bd = te.gb & 3; if ((bd == B_EXACT) || (bd == B_LOWER && ttScore > best) || (bd == B_UPPER && ttScore < best)) best = ttScore; }
      if (best >= beta) { if (!ttHit) TT.store(pos.key, 0, B_LOWER, scoreToTT(best, ply), standPat, 0); return best; }
      if (best > alpha) alpha = best;
    }
    Move ml[256]; int sc[256];
    int n = pos.gen(ml, inCheck ? 0 : 1);
    scoreMoves(ml, sc, n, ttMove, ply, inCheck);
    int moveCount = 0; Move bestMove = 0; int origAlpha = alpha;
    for (int i = 0; i < n; i++) {
      Move m = pickBest(ml, sc, n, i);
      if (!pos.legal(m)) continue;
      moveCount++;
      if (!inCheck) {
        if (!isPromo(m)) {
          int vic = mflag(m) == F_EP ? PAWN : ptOf(pos.board[mto(m)]);
          if (standPat + SEEV[vic] + 200 <= alpha && !pos.givesCheck(m)) continue;
        }
        if (!pos.seeGE(m, 0)) continue;
      } else if (best > -MATE_IN_MAX && moveCount > 1 && !isTactical(m) && !pos.seeGE(m, -50) ) {
        continue;
      }
      ss[ply].move = m; ss[ply].cont = contHist[pos.board[mfrom(m)]][mto(m)];
      pos.make(m);
      int v = -qsearch(-beta, -alpha, ply + 1, isPV);
      pos.unmake();
      if (g_stop) return 0;
      if (v > best) {
        best = v;
        if (v > alpha) {
          bestMove = m; alpha = v;
          if (isPV) { pv[ply][0] = m; for (int k = 0; k < pvLen[ply + 1]; k++) pv[ply][k + 1] = pv[ply + 1][k]; pvLen[ply] = pvLen[ply + 1] + 1; }
          if (v >= beta) break;
        }
      }
    }
    if (inCheck && moveCount == 0) return -MATE + ply;
    TT.store(pos.key, 0, best >= beta ? B_LOWER : (isPV && best > origAlpha ? B_EXACT : B_UPPER), scoreToTT(best, ply), standPat, bestMove);
    return best;
  }

  void updateCont(int ply, int pc, int to, int bonus) {
    for (int k : {1, 2, 4}) if (ply >= k && ss[ply - k].move) hupd(ss[ply - k].cont[pc][to], bonus);
  }

  int search(int depth, int alpha, int beta, int ply, bool isPV, bool cutNode) {
    pvLen[ply] = 0;
    if (depth <= 0) return qsearch(alpha, beta, ply, isPV);
    if ((++nodes & 1023) == 0) checkTime();
    if (g_stop) return 0;
    if (isPV && ply > seldepth) seldepth = ply;
    const bool root = ply == 0;
    const int us = pos.stm;
    if (!root) {
      if (isDrawNode(ply)) return 0;
      if (ply >= MAX_PLY - 1) return evaluate(pos);
      alpha = std::max(alpha, -MATE + ply); beta = std::min(beta, MATE - ply - 1);
      if (alpha >= beta) return alpha;
    }
    const Move excluded = ss[ply].excluded;
    const bool inCheck = pos.inCheck();
    ss[ply].inCheck = inCheck;
    ss[ply + 1].excluded = 0; ss[ply + 2].killers[0] = ss[ply + 2].killers[1] = 0;
    U64 key = pos.key ^ (excluded ? (U64)excluded * 0x9E3779B97F4A7C15ULL : 0);
    TTEntry te; bool ttHit = TT.probe(key, te);
    Move ttMove = ttHit ? te.move : 0; int ttScore = NONE_SC, ttDepth = -1, ttBound = 0;
    if (ttHit) { ttScore = scoreFromTT(te.score, ply); ttDepth = te.depth; ttBound = te.gb & 3; }
    if (root) { if (!ttMove || !ttHit) ttMove = rootHasBest ? rootBest : ttMove; }
    if (!isPV && !excluded && ttHit && ttDepth >= depth && ttScore != NONE_SC && pos.rule50 < 90 &&
        ((ttBound == B_EXACT) || (ttBound == B_LOWER && ttScore >= beta) || (ttBound == B_UPPER && ttScore <= alpha))) {
      if (ttBound == B_LOWER && ttMove && !isTactical(ttMove) && ttScore >= beta && pos.board[mfrom(ttMove)]) {
        // reward a quiet TT move that caused a cutoff
        hupd(mainHist[us][mfrom(ttMove)][mto(ttMove)], std::min(16 * depth * depth, 1500));
      }
      return ttScore;
    }
    int eval;
    if (inCheck) eval = NONE_SC;
    else {
      eval = (ttHit && te.eval != NONE_SC) ? te.eval : evaluate(pos);
      if (ttHit && ttScore != NONE_SC && !excluded) {
        if ((ttBound == B_EXACT) || (ttBound == B_LOWER && ttScore > eval) || (ttBound == B_UPPER && ttScore < eval)) eval = ttScore;
      }
    }
    ss[ply].eval = inCheck ? NONE_SC : (ttHit && te.eval != NONE_SC ? te.eval : eval);
    bool improving = false;
    if (!inCheck) {
      if (ply >= 2 && ss[ply - 2].eval != NONE_SC) improving = ss[ply].eval > ss[ply - 2].eval;
      else if (ply >= 4 && ss[ply - 4].eval != NONE_SC) improving = ss[ply].eval > ss[ply - 4].eval;
    }
    const bool hasPieces = pos.hasPieces(us);

    if (!isPV && !inCheck && !excluded) {
      // razoring
      if (depth <= 3 && eval + 220 + 140 * depth <= alpha) {
        int v = qsearch(alpha - 1, alpha, ply, false);
        if (v < alpha) return v;
      }
      // reverse futility
      if (depth <= 8 && eval - (82 * depth - 55 * improving) >= beta && eval < MATE_IN_MAX && beta > -MATE_IN_MAX)
        return (eval + beta) / 2;
      // null move
      if (depth >= 3 && eval >= beta && ply >= 1 && ss[ply - 1].move != 0 && hasPieces && beta > -MATE_IN_MAX &&
          (ply >= nmpMinPly || us != nmpColor) && (!ttHit || ttBound == B_LOWER || ttScore >= beta || ttScore == NONE_SC)) {
        int R = 3 + depth / 3 + std::min((eval - beta) / 190, 3);
        ss[ply].move = 0; ss[ply].cont = contHist[0][0];
        pos.makeNull();
        int v = -search(depth - R, -beta, -beta + 1, ply + 1, false, !cutNode);
        pos.unmakeNull();
        if (g_stop) return 0;
        if (v >= beta) {
          if (v >= MATE_IN_MAX) v = beta;
          if (depth < 10 || nmpMinPly) return v;
          nmpMinPly = ply + 3 * (depth - R) / 4; nmpColor = us;
          int vv = search(depth - R, beta - 1, beta, ply, false, false);
          nmpMinPly = 0;
          if (g_stop) return 0;
          if (vv >= beta) return v;
        }
      }
      // probcut
      int probBeta = beta + 185 - 50 * improving;
      if (depth >= 5 && std::abs(beta) < MATE_IN_MAX && !(ttHit && ttDepth >= depth - 3 && ttScore != NONE_SC && ttScore < probBeta)) {
        Move ml[256]; int sc[256]; int n = pos.gen(ml, 1); scoreMoves(ml, sc, n, ttMove, ply, false);
        for (int i = 0; i < n; i++) {
          Move m = pickBest(ml, sc, n, i);
          if (m == excluded || !pos.legal(m) || !pos.seeGE(m, probBeta - eval)) continue;
          ss[ply].move = m; ss[ply].cont = contHist[pos.board[mfrom(m)]][mto(m)];
          pos.make(m);
          int v = -qsearch(-probBeta, -probBeta + 1, ply + 1, false);
          if (v >= probBeta) v = -search(depth - 4, -probBeta, -probBeta + 1, ply + 1, false, !cutNode);
          pos.unmake();
          if (g_stop) return 0;
          if (v >= probBeta) { TT.store(key, depth - 3, B_LOWER, scoreToTT(v, ply), ss[ply].eval, m); return v; }
        }
      }
    }
    // internal iterative reduction
    if (depth >= 4 && !ttMove && !inCheck && (isPV || cutNode)) depth--;

    Move ml[256]; int sc[256];
    int n = pos.gen(ml, 0);
    scoreMoves(ml, sc, n, ttMove, ply, inCheck);
    Move quiets[64]; int nq = 0; Move caps[32]; int ncap = 0;
    int best = -INF, moveCount = 0; Move bestMove = 0; int origAlpha = alpha;
    bool skipQuiets = false;
    Move counter = 0;
    if (ply >= 1 && ss[ply - 1].move) { Move pm = ss[ply - 1].move; counter = counterMv[pos.board[mto(pm)]][mto(pm)]; }

    for (int i = 0; i < n; i++) {
      Move m = pickBest(ml, sc, n, i);
      if (m == excluded) continue;
      if (root) { bool ex = false; for (int k = 0; k < nExcl; k++) if (rootExcl[k] == m) ex = true; if (ex) continue; }
      if (!pos.legal(m)) continue;
      const bool tactical = isTactical(m);
      if (skipQuiets && !tactical && m != ttMove) { moveCount++; continue; }
      moveCount++;
      const int pc = pos.board[mfrom(m)], to = mto(m);
      int lmrDepth = std::max(0, depth - LMR[std::min(depth, 63)][std::min(moveCount, 63)]);
      int hs = tactical ? 0 : quietScore(ply, m);
      bool givesChk = false, chkKnown = false;
      if (!root && best > -MATE_IN_MAX && hasPieces) {
        if (!tactical) {
          if (moveCount >= (3 + depth * depth) / (2 - improving) && depth <= 8 && !inCheck) { skipQuiets = true; continue; }
          if (lmrDepth <= 6 && !inCheck && eval != NONE_SC && eval + 95 + 85 * lmrDepth <= alpha) {
            givesChk = pos.givesCheck(m); chkKnown = true;
            if (!givesChk) { skipQuiets = true; continue; }
          }
          if (lmrDepth <= 4 && hs < -3500 * depth && !inCheck) continue;
          if (lmrDepth <= 8 && !pos.seeGE(m, -22 * lmrDepth * lmrDepth - 10 * lmrDepth)) {
            if (!chkKnown) { givesChk = pos.givesCheck(m); chkKnown = true; }
            if (!givesChk) continue;
          }
        } else if (depth <= 6 && !inCheck) {
          if (!pos.seeGE(m, -95 * depth)) continue;
        }
      }
      // extensions
      int ext = 0;
      if (!root && depth >= 8 && m == ttMove && !excluded && ttDepth >= depth - 3 && (ttBound & B_LOWER) &&
          std::abs(ttScore) < MATE_IN_MAX && ttScore != NONE_SC) {
        int sBeta = ttScore - 2 * depth;
        ss[ply].excluded = m;
        int v = search((depth - 1) / 2, sBeta - 1, sBeta, ply, false, cutNode);
        ss[ply].excluded = 0;
        if (g_stop) return 0;
        if (v < sBeta) ext = (!isPV && v < sBeta - 22) ? 2 : 1;
        else if (sBeta >= beta) return sBeta;
        else if (ttScore >= beta) ext = -1;
        else if (cutNode) ext = -1;
      }
      if (ext == 0 && !root) {
        if (!chkKnown) { givesChk = pos.givesCheck(m); chkKnown = true; }
        if (givesChk && (isPV || pos.seeGE(m, 0))) ext = 1;
      } else if (!chkKnown) { givesChk = pos.givesCheck(m); chkKnown = true; }
      if (ply + 1 >= MAX_PLY - 2) ext = std::min(ext, 0);
      int newDepth = depth - 1 + ext;

      ss[ply].move = m; ss[ply].cont = contHist[pc][to];
      int cap = tactical ? (mflag(m) == F_EP ? PAWN : ptOf(pos.board[to])) : 0;
      pos.make(m);
      int v;
      if (depth >= 3 && moveCount > 1 + (root ? 1 : 0) && (!tactical || sc[i] < 0 || !isPV)) {
        int r = LMR[std::min(depth, 63)][std::min(moveCount, 63)];
        if (tactical) r = r * 2 / 3;
        if (isPV) r--;
        if (!improving) r++;
        if (cutNode) r++;
        if (givesChk) r--;
        if (inCheck) r--;
        if (ttMove && isTactical(ttMove)) r++;
        if (!tactical) {
          if (m == ss[ply].killers[0] || m == ss[ply].killers[1] || m == counter) r--;
          r -= std::max(-2, std::min(2, hs / 5500));
        }
        int d = std::max(1, std::min(newDepth - r, newDepth));
        v = -search(d, -alpha - 1, -alpha, ply + 1, false, true);
        if (v > alpha && d < newDepth) v = -search(newDepth, -alpha - 1, -alpha, ply + 1, false, !cutNode);
      } else if (!isPV || moveCount > 1) {
        v = -search(newDepth, -alpha - 1, -alpha, ply + 1, false, !cutNode);
      } else v = alpha + 1;
      if (isPV && (moveCount == 1 || v > alpha)) v = -search(newDepth, -beta, -alpha, ply + 1, true, false);
      pos.unmake();
      if (g_stop) return 0;

      if (v > best) {
        best = v;
        if (v > alpha) {
          bestMove = m; alpha = v;
          if (root && multiPVidx == 0) { rootBest = m; rootScore = v; rootHasBest = true; }
          if (isPV) { pv[ply][0] = m; for (int k = 0; k < pvLen[ply + 1]; k++) pv[ply][k + 1] = pv[ply + 1][k]; pvLen[ply] = pvLen[ply + 1] + 1; }
          if (v >= beta) break;
        }
      }
      if (m != bestMove) { if (tactical) { if (ncap < 32) caps[ncap++] = m; } else if (nq < 64) quiets[nq++] = m; }
      (void)cap;
    }
    if (moveCount == 0) return excluded ? alpha : (inCheck ? -MATE + ply : 0);
    if (best >= beta && bestMove) {
      int bonus = std::min(1700, 330 * depth - 250);
      if (!isTactical(bestMove)) {
        if (ss[ply].killers[0] != bestMove) { ss[ply].killers[1] = ss[ply].killers[0]; ss[ply].killers[0] = bestMove; }
        int pc = pos.board[mfrom(bestMove)], to = mto(bestMove);
        if (ply >= 1 && ss[ply - 1].move) { Move pm = ss[ply - 1].move; (void)pm; }
        if (ply >= 1 && ss[ply - 1].move) counterMv[pos.board[mto(ss[ply - 1].move)]][mto(ss[ply - 1].move)] = bestMove;
        hupd(mainHist[us][mfrom(bestMove)][to], bonus); updateCont(ply, pc, to, bonus);
        for (int k = 0; k < nq; k++) {
          Move q = quiets[k]; int qpc = pos.board[mfrom(q)], qt = mto(q);
          hupd(mainHist[us][mfrom(q)][qt], -bonus); updateCont(ply, qpc, qt, -bonus);
        }
      } else {
        int vic = mflag(bestMove) == F_EP ? PAWN : ptOf(pos.board[mto(bestMove)]);
        hupd(capHist[pos.board[mfrom(bestMove)]][mto(bestMove)][vic], bonus);
      }
      for (int k = 0; k < ncap; k++) {
        Move c = caps[k]; int vic = mflag(c) == F_EP ? PAWN : ptOf(pos.board[mto(c)]);
        hupd(capHist[pos.board[mfrom(c)]][mto(c)][vic], -bonus);
      }
    }
    if (!excluded && !(root && multiPVidx > 0))
      TT.store(key, depth, best >= beta ? B_LOWER : (isPV && bestMove ? B_EXACT : B_UPPER), scoreToTT(best, ply), ss[ply].eval, bestMove);
    return best;
  }

  static std::string scoreStr(int s) {
    if (s >= MATE_IN_MAX) return "mate " + std::to_string((MATE - s + 1) / 2);
    if (s <= -MATE_IN_MAX) return "mate -" + std::to_string((MATE + s + 1) / 2);
    return "cp " + std::to_string(s);
  }

  void go(const Limits &lim) {
    g_stop = false; g_searching = true;
    t0 = nowTP(); nodes = 0; seldepth = 0; nmpMinPly = 0; rootHasBest = false; rootBest = 0;
    TT.newSearch();
    for (int i = 0; i < MAX_PLY + 12; i++) { stackBuf[i].move = 0; stackBuf[i].eval = NONE_SC; stackBuf[i].killers[0] = stackBuf[i].killers[1] = 0; stackBuf[i].excluded = 0; stackBuf[i].cont = contHist[0][0]; stackBuf[i].inCheck = false; }
    Move legalMoves[256]; int nLegal = pos.genLegal(legalMoves);
    if (nLegal == 0) { say("bestmove 0000"); g_searching = false; return; }
    // time management
    timed = false; hardMs = softMs = 0; nodeLimit = (U64)lim.nodes;
    long long myTime = pos.stm == WHITE ? lim.wtime : lim.btime, myInc = pos.stm == WHITE ? lim.winc : lim.binc;
    if (lim.movetime > 0) { timed = true; hardMs = std::max(1LL, lim.movetime - 5); softMs = hardMs * 7 / 10; }
    else if (myTime >= 0 && !lim.infinite) {
      timed = true;
      long long avail = std::max(1LL, myTime - OPT_OVERHEAD);
      int mtg = lim.movestogo > 0 ? std::min(lim.movestogo, 50) : 28;
      long long opt = avail / mtg + myInc * 3 / 4;
      opt = std::min(opt, avail * 45 / 100); opt = std::max(1LL, opt);
      softMs = opt; hardMs = std::min(avail * 80 / 100, opt * 4); hardMs = std::max(1LL, hardMs);
    }
    int maxDepth = lim.depth > 0 ? std::min(lim.depth, MAX_PLY - 8) : MAX_PLY - 8;
    int mpv = std::max(1, std::min(OPT_MULTIPV, nLegal));
    bool single = nLegal == 1 && !lim.infinite && lim.depth == 0 && lim.nodes == 0;
    if (single) maxDepth = std::min(maxDepth, 6);
    Move prevBest = 0; int stable = 0, prevScore = 0; Move lastBest = 0; int lastDepthDone = 0;
    Move finalPV[MAX_PLY + 2]; int finalLen = 0;
    for (int depth = 1; depth <= maxDepth; depth++) {
      nExcl = 0;
      Move linePV[64][MAX_PLY + 2]; int lineLen[64]; int lineScore[64]; int lines = 0;
      bool aborted = false;
      for (int pi = 0; pi < mpv; pi++) {
        multiPVidx = pi;
        int alpha = -INF, beta = INF, delta = 16;
        int center = (depth >= 5 && pi == 0 && depth > 1) ? prevScore : 0;
        if (depth >= 5) { alpha = std::max(-INF, center - delta); beta = std::min(INF, center + delta); }
        if (depth >= 5 && pi > 0) { alpha = -INF; beta = INF; }
        int score = 0;
        for (;;) {
          score = search(depth, alpha, beta, 0, true, false);
          if (g_stop) { aborted = true; break; }
          if (score <= alpha && alpha > -INF) { beta = (alpha + beta) / 2; alpha = std::max(-INF, score - delta); delta += delta / 2 + 5; if (delta > 600) alpha = -INF; }
          else if (score >= beta && beta < INF) { beta = std::min(INF, score + delta); delta += delta / 2 + 5; if (delta > 600) beta = INF; }
          else break;
        }
        if (aborted) break;
        if (pvLen[0] == 0) { pv[0][0] = legalMoves[0]; pvLen[0] = 1; }
        memcpy(linePV[pi], pv[0], sizeof(Move) * (pvLen[0])); lineLen[pi] = pvLen[0]; lineScore[pi] = score; lines++;
        rootExcl[nExcl++] = pv[0][0];
      }
      if (lines == 0) break;
      // completed (at least partially) iteration — publish
      if (!aborted || lines > 0) {
        lastDepthDone = depth;
        for (int pi = 0; pi < lines; pi++) {
          std::string s = "info depth " + std::to_string(depth) + " seldepth " + std::to_string(seldepth) + " multipv " + std::to_string(pi + 1) +
                          " score " + scoreStr(lineScore[pi]) + " nodes " + std::to_string(nodes) + " nps " + std::to_string(msSince(t0) > 0 ? nodes * 1000 / msSince(t0) : nodes) +
                          " hashfull " + std::to_string(TT.hashfull()) + " time " + std::to_string(msSince(t0)) + " pv";
          for (int k = 0; k < lineLen[pi]; k++) s += " " + moveStr(linePV[pi][k]);
          say(s);
        }
        if (lines == mpv && !aborted) { memcpy(finalPV, linePV[0], sizeof(Move) * lineLen[0]); finalLen = lineLen[0]; }
        if (lines == mpv) { rootBest = linePV[0][0]; rootHasBest = true; }
      }
      if (aborted) break;
      Move cur = linePV[0][0];
      stable = (cur == prevBest) ? stable + 1 : 0; prevBest = cur;
      int sc0 = lineScore[0];
      if (timed && lim.movetime <= 0) {
        double f = stable >= 4 ? 0.65 : stable >= 2 ? 0.85 : 1.0;
        if (depth >= 6 && sc0 < prevScore - 25) f *= 1.35;
        if (depth >= 6 && cur != lastBest && lastBest) f *= 1.25;
        if (msSince(t0) > (long long)(softMs * f)) break;
      } else if (timed && msSince(t0) > softMs) break;
      if (single && depth >= 6) break;
      if (std::abs(sc0) >= MATE_IN_MAX && depth >= 8 && (MATE - std::abs(sc0)) * 2 < depth && !lim.infinite && lim.depth == 0) break;
      lastBest = cur; prevScore = sc0;
      if (g_stop) break;
    }
    (void)finalLen; (void)lastDepthDone;
    Move bm = rootHasBest ? rootBest : legalMoves[0];
    // safety: best move must be legal
    bool ok = false; for (int i = 0; i < nLegal; i++) if (legalMoves[i] == bm) ok = true;
    if (!ok) bm = legalMoves[0];
    std::string out = "bestmove " + moveStr(bm);
    // ponder move from the stored PV if it is legal
    if (finalLen >= 2 && finalPV[0] == bm) {
      Position q = pos; q.make(bm); if (q.parseMove(moveStr(finalPV[1]))) out += " ponder " + moveStr(finalPV[1]);
    }
    g_searching = false;
    say(out);
  }
};

static Searcher *S = nullptr;

// =====================================================================
//                                  UCI
// =====================================================================
static const char *START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
#ifdef __EMSCRIPTEN__
static int OPT_HASH = 32;
#else
static int OPT_HASH = 64;
#endif
static bool g_ready = false;

static void engineInit() {
  if (g_ready) return;
  initTables(); initEval();
  for (int d = 0; d < 64; d++) for (int m = 0; m < 64; m++)
    LMR[d][m] = (d == 0 || m == 0) ? 0 : (int)(0.75 + std::log((double)d) * std::log((double)m) / 2.25);
  S = new Searcher(); S->clearHistory();
  S->pos.setFen(START_FEN);
  TT.resize(OPT_HASH);
  g_ready = true;
}

#ifndef __EMSCRIPTEN__
static std::thread g_thread;
static void joinSearch() { if (g_thread.joinable()) g_thread.join(); }
static void stopSearch() { g_stop = true; joinSearch(); }
#else
static void joinSearch() {}
static void stopSearch() { g_stop = true; }
#endif

static void doPosition(std::istringstream &is) {
  std::string tok, fen;
  is >> tok;
  Position np; bool ok = true;
  if (tok == "startpos") { ok = np.setFen(START_FEN); is >> tok; }
  else if (tok == "fen") {
    while (is >> tok && tok != "moves") fen += tok + " ";
    ok = np.setFen(fen);
  } else return;
  if (!ok) { say("info string invalid position"); return; }
  if (tok == "moves") {
    std::string ms;
    while (is >> ms) {
      Move m = np.parseMove(ms);
      if (!m) { say("info string illegal move " + ms); break; }
      np.make(m);
    }
  }
  S->pos = np;
}

static void doGo(std::istringstream &is) {
  Limits lim; std::string t;
  while (is >> t) {
    if (t == "wtime") is >> lim.wtime; else if (t == "btime") is >> lim.btime;
    else if (t == "winc") is >> lim.winc; else if (t == "binc") is >> lim.binc;
    else if (t == "movetime") is >> lim.movetime; else if (t == "depth") is >> lim.depth;
    else if (t == "nodes") is >> lim.nodes; else if (t == "movestogo") is >> lim.movestogo;
    else if (t == "infinite") lim.infinite = true;
    else if (t == "ponder") {}
  }
  stopSearch();
#ifndef __EMSCRIPTEN__
  g_thread = std::thread([lim]() { S->go(lim); });
#else
  S->go(lim);
#endif
}

static const char *BENCH_FENS[] = {
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
  "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
  "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
  "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
  "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 0 10",
  "8/8/4k3/8/2p5/8/B2K4/8 w - - 0 1",
  "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
  "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"};

static void doBench(int depth) {
  U64 total = 0; auto t = nowTP();
  for (const char *f : BENCH_FENS) {
    TT.clear(); S->clearHistory(); S->pos.setFen(f);
    Limits l; l.depth = depth;
    // silent search: redirect info output by running search then reading nodes
    fflush(stdout);
    S->go(l); total += S->nodes;
  }
  long long ms = msSince(t); if (ms < 1) ms = 1;
  say("Nodes searched: " + std::to_string(total));
  say("Nodes/second: " + std::to_string(total * 1000 / ms));
}

static void handle(const std::string &line, bool &quit) {
  std::istringstream is(line); std::string cmd; is >> cmd;
  if (cmd == "uci") {
    say("id name Atelier 4"); say("id author Atelier");
    say("option name Hash type spin default 64 min 1 max 4096");
    say("option name MultiPV type spin default 1 min 1 max 64");
    say("option name Move Overhead type spin default 30 min 0 max 5000");
    say("option name Clear Hash type button");
    say("uciok");
  } else if (cmd == "isready") { stopSearch(); say("readyok"); }
  else if (cmd == "ucinewgame") { stopSearch(); TT.clear(); S->clearHistory(); S->pos.setFen(START_FEN); }
  else if (cmd == "setoption") {
    std::string t, name, value; is >> t; // "name"
    while (is >> t && t != "value") name += (name.empty() ? "" : " ") + t;
    while (is >> t) value += (value.empty() ? "" : " ") + t;
    stopSearch();
    if (name == "Hash") { OPT_HASH = std::max(1, std::min(4096, atoi(value.c_str()))); TT.resize(OPT_HASH); }
    else if (name == "MultiPV") OPT_MULTIPV = std::max(1, std::min(64, atoi(value.c_str())));
    else if (name == "Move Overhead") OPT_OVERHEAD = std::max(0, std::min(5000, atoi(value.c_str())));
    else if (name == "Clear Hash") { TT.clear(); S->clearHistory(); }
  }
  else if (cmd == "position") { stopSearch(); doPosition(is); }
  else if (cmd == "go") doGo(is);
  else if (cmd == "stop") stopSearch();
  else if (cmd == "quit") { stopSearch(); quit = true; }
  else if (cmd == "d") { say(S->pos.fen()); }
  else if (cmd == "eval") { say("eval " + std::to_string(evaluate(S->pos))); }
  else if (cmd == "perft") {
    int d = 1; is >> d; auto t = nowTP(); U64 n = perft(S->pos, d); long long ms = std::max(1LL, msSince(t));
    say("perft " + std::to_string(d) + " = " + std::to_string(n) + " (" + std::to_string(ms) + " ms)");
  }
  else if (cmd == "bench") { int d = 10; is >> d; stopSearch(); doBench(d); }
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
extern "C" {
EMSCRIPTEN_KEEPALIVE void atelier_cmd(const char *line) { engineInit(); bool q = false; handle(line, q); }
}
int main() { engineInit(); return 0; }
#else
int main(int argc, char **argv) {
  engineInit();
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc > 1 && std::string(argv[1]) == "bench") { int d = argc > 2 ? atoi(argv[2]) : 10; doBench(d); return 0; }
  std::string line; bool quit = false;
  while (!quit && std::getline(std::cin, line)) handle(line, quit);
  joinSearch();
  return 0;
}
#endif
