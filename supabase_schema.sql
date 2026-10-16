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
