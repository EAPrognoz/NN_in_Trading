//+------------------------------------------------------------------+
//|                                                   MyTreeNode.mqh |
//|                                                       AssocRules |
//|                                              Copyright 2022, DNG |
//|                           https://www.mql5.com/ru/articles/11141 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//---
#include <Arrays\ArrayObj.mqh>
//---
//+------------------------------------------------------------------+
//| CMyTreeNode                                                      |
//+------------------------------------------------------------------+
class CMyTreeNode : public CArrayObj
  {
protected:
   CMyTreeNode *m_cParent;
   ulong       m_iIndex;
   float       m_dSupport;

public:
                     CMyTreeNode();
                    ~CMyTreeNode();
//--- methods of access to protected data
//+------------------------------------------------------------------+
//| Parent                                                           |
//+------------------------------------------------------------------+
   CMyTreeNode*      Parent(void)           const
     {
      return(m_cParent);
     }
//+------------------------------------------------------------------+
//| Parent                                                           |
//+------------------------------------------------------------------+
   void              Parent(CMyTreeNode *node)
     {
      m_cParent=node;
     }
//+------------------------------------------------------------------+
//| IncreaseSupport                                                  |
//+------------------------------------------------------------------+
   void              IncreaseSupport(float support)
     {
      m_dSupport += support;
     }
//+------------------------------------------------------------------+
//| GetSupport                                                       |
//+------------------------------------------------------------------+
   float            GetSupport(void)
     {
      return(m_dSupport);
     }
//+------------------------------------------------------------------+
//| SetSupport                                                       |
//+------------------------------------------------------------------+
   void              SetSupport(float support)
     {
      m_dSupport=support;
     }
//+------------------------------------------------------------------+
//| ID                                                               |
//+------------------------------------------------------------------+
   ulong             ID(void)
     {
      return(m_iIndex);
     }
//+------------------------------------------------------------------+
//| ID                                                               |
//+------------------------------------------------------------------+
   void              ID(ulong ID)
     {
      m_iIndex=ID;
     }
   float            GetConfidence(void);
   float            GetConfidence(const ulong ID);
   //---
   CMyTreeNode      *GetNext(const ulong ID);
   matrix            NodesGetConfidence(void);
   CMyTreeNode      *AddNode(const ulong ID,float weight=0);
   bool              DeleteNode(const ulong ID);
   //---
   uint              GetDepth(void);
   //---
   bool              Mining(vectorf &data,matrixf &paths,const ulong ID,float min_conf);

//--- methods for working with files
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);

//--- method of creating an element of array
   virtual bool      CreateElement(const int index);
  };
//+------------------------------------------------------------------+
//| m d Su pp or t implementation                                    |
//+------------------------------------------------------------------+
                              CMyTreeNode::CMyTreeNode() : m_dSupport(0)
  {
   Clear();
  }
//+------------------------------------------------------------------+
//| ~CMyTreeNode - destructor                                        |
//+------------------------------------------------------------------+
CMyTreeNode::~CMyTreeNode()
  {
  }
//+------------------------------------------------------------------+
//| GetConfidence                                                    |
//+------------------------------------------------------------------+
float CMyTreeNode::GetConfidence(void)
  {
   CMyTreeNode *parent=Parent();
   if(!parent)
      return(1);
//---
   float result=m_dSupport / parent.GetSupport();

   return(result);
  }
//+------------------------------------------------------------------+
//| GetConfidence                                                    |
//+------------------------------------------------------------------+
float CMyTreeNode::GetConfidence(const ulong ID)
  {
   CMyTreeNode *temp=GetNext(ID);
   if(!temp)
      return(0);
//---
   return(temp.GetConfidence());
  }
//+------------------------------------------------------------------+
//| GetNext - getting the element by ID                              |
//+------------------------------------------------------------------+
CMyTreeNode *CMyTreeNode::GetNext(const ulong ID)
  {
   if(m_data_total<=0)
      return(NULL);
//---
   CMyTreeNode *result=NULL;
   for(int i=0; i<m_data_total; i++)
     {
      CMyTreeNode *temp=m_data[i];
      if(!temp)
         continue;
      if(temp.ID()!=ID)
         continue;
      result=temp;
      break;
     }
//---
   return(result);
  }
//+------------------------------------------------------------------+
//| DeleteNode                                                       |
//+------------------------------------------------------------------+
bool CMyTreeNode::DeleteNode(const ulong ID)
  {
   for(int i=0; i<m_data_total; i++)
     {
      CMyTreeNode *temp=m_data[i];
      if(!temp)
        {
         if(!Delete(i))
            continue;

         return(DeleteNode(ID));
        }
      if(temp.ID()!=ID)
         continue;

      return(Delete(i));
     }
//---
   return(false);
  }
//+------------------------------------------------------------------+
//| AddNode - adding a node                                          |
//+------------------------------------------------------------------+
CMyTreeNode *CMyTreeNode::AddNode(const ulong ID,float support)
  {
   CMyTreeNode *node=new CMyTreeNode();
   if(!node)
      return(node);

   node.ID(ID);
   if(!Add(node))
     {
      delete node;

      return(node);
     }
   node.Parent(GetPointer(this));
//---
   if(support>0)
      node.SetSupport(support);

   return(node);
  }
//+------------------------------------------------------------------+
//| GetDepth                                                         |
//+------------------------------------------------------------------+
uint CMyTreeNode::GetDepth(void)
  {
   CMyTreeNode *parent=Parent();
   if(!parent)
      return(0);

   return(parent.GetDepth()+1);
  }
//+------------------------------------------------------------------+
//| Mining                                                           |
//+------------------------------------------------------------------+
bool CMyTreeNode::Mining(vectorf &supports,matrixf &paths,const ulong ID,float min_conf)
  {
   if(ID==m_iIndex)
      if(GetConfidence()<min_conf)
         return(true);
//---
   float support=m_dSupport;
   for(int i=0; i<m_data_total; i++)
     {
      CMyTreeNode *temp=m_data[i];
      if(!temp)
        {
         if(Delete(i))
            i--;
         continue;
        }
      if(!temp.Mining(supports,paths,(ID==m_iIndex ? ULONG_MAX : ID),min_conf))
         return(false);

      support -= temp.GetSupport();
      if(temp.ID()==ID)
         if(Delete(i))
            i--;
     }
//---
   if(ID==m_iIndex||ID==ULONG_MAX)
      if(support>0&&!!m_cParent)
        {
         CMyTreeNode *parent=m_cParent;
         ulong       row=paths.Rows();
         if(!paths.Resize(row+1,paths.Cols()))
            return(false);

         if(!paths.Row(vectorf::Zeros(paths.Cols()),row))
            return(false);

         supports[m_iIndex] += support;
         while(!!parent)
           {
            if(parent.ID()!=ULONG_MAX)
              {
               supports[parent.ID()] += support;
               paths[row,parent.ID()]=support;
              }
            parent=parent.Parent();
           }
        }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| CreateElement                                                    |
//+------------------------------------------------------------------+
bool CMyTreeNode::CreateElement(const int index)
  {
   if(index>=m_data_max)
      if(!Resize(index+1))
         return(false);

   m_data[index]=new CMyTreeNode();
   if(!m_data[index])
      return(false);
   m_data_total=fmax(index,m_data_total);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Save                                                             |
//+------------------------------------------------------------------+
bool CMyTreeNode::Save(const int file_handle)
  {
   if(!CArrayObj::Save(file_handle))
      return(false);

   if(FileWriteLong(file_handle,m_iIndex)<sizeof(m_iIndex))
      return(false);
   if(FileWriteDouble(file_handle,m_dSupport)<sizeof(m_dSupport))
      return(false);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Load                                                             |
//+------------------------------------------------------------------+
bool CMyTreeNode::Load(const int file_handle)
  {
   if(!CArrayObj::Load(file_handle))
      return(false);

   m_iIndex=FileReadLong(file_handle);
   m_dSupport=(float)FileReadDouble(file_handle);
   CMyTreeNode *point=GetPointer(this);
//---
   for(int i=0; i<m_data_total; i++)
     {
      CMyTreeNode *temp=m_data[i];
      if(!temp)
        {
         if(Delete(i))
            i--;
         continue;
        }
      temp.Parent(point);
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+

