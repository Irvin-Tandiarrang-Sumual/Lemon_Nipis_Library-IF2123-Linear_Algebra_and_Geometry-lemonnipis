import { notFound } from "next/navigation";
import mapperData from "@/data/mapper.json";
import { BookDetailWrapper } from "@/components/book-detail/detail-wrapper";

async function getBookById(id: string) {
  const data = mapperData as Record<string, any>;
  const book = data[id];
  if (!book) return null;
  return { id, ...book };
}

export default async function BookDetailPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const book = await getBookById(id);

  if (!book) {
    notFound();
  }

  return (
    <BookDetailWrapper book={book} />
  );
}