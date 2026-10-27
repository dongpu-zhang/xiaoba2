-- 小巴2.0 对话持久化表（复用 book-reader 的 Supabase 项目，不影响 books/settings 表）
create table if not exists xiaoba_conversations (
  id uuid primary key default gen_random_uuid(),
  title text not null default '新对话',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists xiaoba_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references xiaoba_conversations(id) on delete cascade,
  role text not null check (role in ('user','assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_xiaoba_messages_conv
  on xiaoba_messages(conversation_id, created_at);

-- WhatsApp 对接：每个手机号绑定一条固定的 conversation，收到消息时按手机号查/建
alter table xiaoba_conversations add column if not exists wa_phone text;
create unique index if not exists idx_xiaoba_conversations_wa_phone
  on xiaoba_conversations(wa_phone) where wa_phone is not null;

-- 微信公众号对接：每个用户 openid 绑定一条固定的 conversation
alter table xiaoba_conversations add column if not exists wx_openid text;
create unique index if not exists idx_xiaoba_conversations_wx_openid
  on xiaoba_conversations(wx_openid) where wx_openid is not null;

-- Telegram 对接：每个 chat_id 绑定一条固定的 conversation
alter table xiaoba_conversations add column if not exists tg_chat_id text;
create unique index if not exists idx_xiaoba_conversations_tg_chat_id
  on xiaoba_conversations(tg_chat_id) where tg_chat_id is not null;
