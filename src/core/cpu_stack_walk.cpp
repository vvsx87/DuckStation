// SPDX-FileCopyrightText: 2012 PPSSPP Project, 2014-2014 PCSX2 Dev Team, 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-2.0+)

#include "cpu_stack_walk.h"
#include "bus.h"
#include "cpu_core_private.h"

#define _RS ((rawOp >> 21) & 0x1F)
#define _RT ((rawOp >> 16) & 0x1F)
#define _RD ((rawOp >> 11) & 0x1F)
#define _IMM16 ((signed short)(rawOp & 0xFFFF))
#define MIPS_REG_SP 29
#define MIPS_REG_FP 30
#define MIPS_REG_RA 31

#define INVALIDTARGET 0xFFFFFFFF

#define MIPSTABLE_IMM_MASK 0xFC000000
#define MIPSTABLE_SPECIAL_MASK 0xFC00003F

namespace MipsStackWalk {
// In the worst case, we scan this far above the pc for an entry.
//constexpr int MAX_FUNC_SIZE = 32768 * 4;
constexpr int MAX_FUNC_SIZE = 1024 * 4;
// After this we assume we're stuck.
constexpr size_t MAX_DEPTH = 1024;

static bool IsSWInstr(u32 rawOp)
{
  return (rawOp & MIPSTABLE_IMM_MASK) == 0xAC000000;
}

static bool IsAddImmInstr(u32 rawOp)
{
  return (rawOp & MIPSTABLE_IMM_MASK) == 0x20000000 || (rawOp & MIPSTABLE_IMM_MASK) == 0x24000000;
}

static bool IsMovRegsInstr(u32 rawOp)
{
  if ((rawOp & MIPSTABLE_SPECIAL_MASK) == 0x00000021)
  {
    return _RS == 0 || _RT == 0;
  }
  return false;
}

static bool IsValidAddress(u32 addr)
{
  return (addr & CPU::PHYSICAL_MEMORY_ADDRESS_MASK) < Bus::RAM_2MB_SIZE;
}

static bool ScanForAllocaSignature(u32 pc)
{
  // In God Eater Burst, for example, after 0880E750, there's what looks like an alloca().
  // It's surrounded by "mov fp, sp" and "mov sp, fp", which is unlikely to be used for other reasons.

  // It ought to be pretty close.
  u32 stop = pc - 32 * 4;
  for (; IsValidAddress(pc) && pc >= stop; pc -= 4)
  {
    u32 rawOp;
    if (!CPU::SafeReadMemoryWord(pc, &rawOp))
      return false;

    // We're looking for a "mov fp, sp" close by a "addiu sp, sp, -N".
    if (IsMovRegsInstr(rawOp) && _RD == MIPS_REG_FP && (_RS == MIPS_REG_SP || _RT == MIPS_REG_SP))
    {
      return true;
    }
  }
  return false;
}

static bool ScanForEntry(StackFrame& frame, u32 entry, u32& ra)
{
  // Let's hope there are no > 1MB functions on the PSP, for the sake of humanity...
  const u32 LONGEST_FUNCTION = 1024 * 1024;
  // TODO: Check if found entry is in the same symbol?  Might be wrong sometimes...

  int ra_offset = -1;
  const u32 start = frame.pc;
  u32 stop = entry;
  if (entry == INVALIDTARGET)
  {
    stop = 0x80000;
  }
  if (stop < start - LONGEST_FUNCTION)
  {
    stop = (LONGEST_FUNCTION > start) ? 0 : (start - LONGEST_FUNCTION);
  }
  for (u32 pc = start; IsValidAddress(pc) && pc >= stop; pc -= 4)
  {
    u32 rawOp;
    if (!CPU::SafeReadMemoryWord(pc, &rawOp))
      return false;

    // Here's where they store the ra address.
    if (IsSWInstr(rawOp) && _RT == MIPS_REG_RA && _RS == MIPS_REG_SP)
    {
      ra_offset = _IMM16;
    }

    if (IsAddImmInstr(rawOp) && _RT == MIPS_REG_SP && _RS == MIPS_REG_SP)
    {
      // A positive imm either means alloca() or we went too far.
      if (_IMM16 > 0)
      {
        // TODO: Maybe check for any alloca() signature and bail?
        continue;
      }
      if (ScanForAllocaSignature(pc))
      {
        continue;
      }

      frame.entry = pc;
      frame.stackSize = -_IMM16;
      if (ra_offset != -1 && IsValidAddress(frame.sp + ra_offset))
      {
        CPU::SafeReadMemoryWord(frame.sp + ra_offset, &ra);
      }
      return true;
    }
  }
  return false;
}

static bool DetermineFrameInfo(StackFrame& frame, u32 possibleEntry, u32 threadEntry, u32& ra)
{
  if (ScanForEntry(frame, possibleEntry, ra))
  {
    // Awesome, found one that looks right.
    return true;
  }
  else if (ra != INVALIDTARGET && possibleEntry != INVALIDTARGET)
  {
    // Let's just assume it's a leaf.
    frame.entry = possibleEntry;
    frame.stackSize = 0;
    return true;
  }

  // Okay, we failed to get one.  Our possibleEntry could be wrong, it often is.
  // Let's just scan upward.
  u32 newPossibleEntry = frame.pc > threadEntry ? threadEntry : frame.pc - MAX_FUNC_SIZE;
  return ScanForEntry(frame, newPossibleEntry, ra);
}

std::vector<StackFrame> Walk(u32 pc, u32 ra, u32 sp, u32 threadEntry, u32 threadStackTop)
{
  std::vector<StackFrame> frames;
  StackFrame current;
  current.pc = pc;
  current.sp = sp;
  current.entry = INVALIDTARGET;
  current.stackSize = -1;

  u32 prevEntry = INVALIDTARGET;
  while (pc != threadEntry)
  {
    u32 possibleEntry = INVALIDTARGET; // GuessEntry(cpu, current.pc);
    if (DetermineFrameInfo(current, possibleEntry, threadEntry, ra))
    {
      frames.push_back(current);
      if (current.entry == threadEntry /*|| GuessEntry(cpu, current.entry) == threadEntry*/)
      {
        break;
      }
      if (current.entry == prevEntry || frames.size() >= MAX_DEPTH)
      {
        // Recursion, means we're screwed.  Let's just give up.
        break;
      }
      prevEntry = current.entry;

      current.pc = ra;
      current.sp += current.stackSize;
      ra = INVALIDTARGET;
      current.entry = INVALIDTARGET;
      current.stackSize = -1;
    }
    else
    {
      // Well, we got as far as we could.
      current.entry = possibleEntry;
      current.stackSize = 0;
      frames.push_back(current);
      break;
    }
  }

  return frames;
}
}; // namespace MipsStackWalk
