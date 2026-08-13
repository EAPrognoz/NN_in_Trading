//+------------------------------------------------------------------+
//| Cell.mqh |
//| Copyright DNG® |
//| https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
#define                    HistoryBars  20            //Depth of history
#define                    Buffer_Size  2112
#define                    FileName     "GoExploer"
//+------------------------------------------------------------------+
//| Cell                                                             |
//+------------------------------------------------------------------+
struct Cell
  {
   int               actions[Buffer_Size];
   float             state[HistoryBars * 12 + 9];
   int               total_actions;
   float             value;
//---
                     Cell(void);
//---
   bool              Save(int file_handle);
   bool              Load(int file_handle);
  };
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Creates and manages object lifecycle for Cell.                 |
//+------------------------------------------------------------------+
Cell::Cell(void)
  {
   ArrayInitialize(actions, -1);
   ArrayInitialize(state, 0);
   value = 0;
   total_actions = 0;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Saves .                                                        |
//+------------------------------------------------------------------+
bool Cell::Save(int file_handle)
  {
   if(file_handle <= 0)
      return false;
   if(FileWriteInteger(file_handle, 999) < INT_VALUE)
      return false;
   if(FileWriteFloat(file_handle, value) < sizeof(float))
      return false;
   if(FileWriteInteger(file_handle, total_actions) < INT_VALUE)
      return false;
   for(int i = 0; i < total_actions; i++)
      if(FileWriteInteger(file_handle, actions[i]) < INT_VALUE)
         return false;
   int size = ArraySize(state);
   if(FileWriteInteger(file_handle, size) < INT_VALUE)
      return false;
   for(int i = 0; i < size; i++)
      if(FileWriteFloat(file_handle, state[i]) < sizeof(float))
         return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| | Loads .                                                        |
//+------------------------------------------------------------------+
bool Cell::Load(int file_handle)
  {
   if(file_handle <= 0)
      return false;
   if(FileReadInteger(file_handle) != 999)
      return false;
   value = FileReadFloat(file_handle);
   total_actions = FileReadInteger(file_handle);
   if(total_actions > Buffer_Size)
      return false;
   for(int i = 0; i < total_actions; i++)
      actions[i] = FileReadInteger(file_handle);
   int size = FileReadInteger(file_handle);
   if(size != (HistoryBars * 12 + 9))
      return false;
   for(int i = 0; i < size; i++)
      state[i] = FileReadFloat(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+

