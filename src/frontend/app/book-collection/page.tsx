import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Image } from "@heroui/image";
import Link from "next/link";
import { BottomPagination } from "@/components/bottom-pagination"; 
import mapperData from "@/data/mapper.json"; 

interface Book {
  id: string;
  title: string;
  cover: string; 
  txt: string;
}

async function getBooks(): Promise<Book[]> {
  const books: Book[] = Object.entries(mapperData).map(([key, value]) => ({
    id: key,
    ...(value as any),
  }));
  return books;
}

export default async function BookCollectionPage({
  searchParams,
}: {
  searchParams: { page?: string };
}) {
  const allBooks = await getBooks();
  const ITEMS_PER_PAGE = 15; 
  const totalItems = allBooks.length;
  const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);
  const currentPage = Number(searchParams.page) || 1;
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentBooks = allBooks.slice(startIndex, endIndex);

  return (
    <section className="w-screen relative left-[50%] right-[50%] -ml-[50vw] -mr-[50vw] py-8">
      
      <div className="w-full px-6 md:px-12"> 
        
        <div className="w-full px-6 md:px-12 mb-10">
            <h1 className="text-3xl font-bold text-center">Book Collection ({totalItems})</h1>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-6 w-full px-6 md:px-12">
          {currentBooks.map((item) => {
            let cleanCover = item.cover ? item.cover.replace(/\\/g, '/') : 'placeholder.jpg';
            const imageUrl = cleanCover.startsWith('http') || cleanCover.startsWith('/')
              ? cleanCover
              : `/${cleanCover}`;

            return (
              <Card
                key={item.id}
                isBlurred
                className="border-none bg-background/60 dark:bg-default-100/50 w-full hover:bg-content2/50 transition-all duration-300"
                shadow="sm"
              >
                <CardBody className="p-0 overflow-hidden">
                  <div className="flex flex-row h-full">
                    
                    {/* BAGIAN GAMBAR (KIRI - FIXED WIDTH)
                        w-32 atau w-40 memastikan gambar tidak terlalu lebar, sisa ruang buat teks.
                    */}
                    <div className="w-32 sm:w-40 h-full relative flex-shrink-0">
                        <Image
                            alt={item.title}
                            className="object-cover w-full h-full rounded-none"
                            src={imageUrl}
                            radius="none"
                            width="100%"
                            height="100%"
                            removeWrapper
                        />
                    </div>

                    {/* BAGIAN KONTEN (KANAN - FLEX GROW) */}
                    <div className="flex flex-col justify-between p-4 flex-grow">
                      <div>
                        <div className="flex justify-between items-start w-full">
                           <div className="flex flex-col pr-2">
                              <span className="text-[10px] uppercase font-bold tracking-widest text-default-500 mb-1">
                                E-Book • #{item.id}
                              </span>
                              <Link href={`/book-collection/${item.id}`} className="group">
                                <h3 className="text-lg sm:text-xl font-bold text-foreground leading-tight line-clamp-2 group-hover:text-primary transition-colors" title={item.title}>
                                  {item.title}
                                </h3>
                              </Link>
                           </div>
                           <Button isIconOnly radius="full" size="sm" variant="light" className="-mr-2 -mt-2 text-default-400">
                           </Button>
                        </div>
                        <p className="text-small text-default-500 line-clamp-2 mt-2 hidden sm:block">
                           {item.txt ? item.txt.substring(0, 100) : "Klik detail untuk membaca sinopsis lengkap buku ini..."}
                        </p>
                      </div>

                      <div className="mt-4 w-full flex justify-end">
                         <Button
                            as={Link}
                            href={`/book-collection/${item.id}`}
                            className="font-medium w-full sm:w-auto px-6"
                            color="primary"
                            radius="full"
                            size="sm"
                            variant="flat"
                         >
                            Baca Detail
                         </Button>
                      </div>
                    </div>
                  </div>
                </CardBody>
              </Card>
            );
          })}
        </div>

        {totalPages > 1 && (
          <div className="mt-12 flex justify-center w-full">
            <BottomPagination totalPages={totalPages} />
          </div>
        )}
      </div>
    </section>
  );
}