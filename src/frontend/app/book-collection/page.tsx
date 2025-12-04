import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Image } from "@heroui/image";
import Link from "next/link";
import { BottomPagination } from "@/components/bottom-pagination"; 

const API_BASE_URL = "http://localhost:8000";

interface Book {
  id: string;
  title: string;
  cover: string; 
  txt: string;
}

interface ApiResponse {
  total: number;
  books: Book[];
}
async function fetchBooks(page: number, query?: string): Promise<ApiResponse> {
  const limit = 15;
  const skip = (page - 1) * limit; 
  
  try {
    let url = "";
    
    if (query) {
      url = `${API_BASE_URL}/api/books/search?title_query=${encodeURIComponent(query)}&skip=${skip}&limit=${limit}`;
    } else {
      url = `${API_BASE_URL}/api/books?skip=${skip}&limit=${limit}`;
    }

    const res = await fetch(url, { cache: "no-store" });
    
    if (!res.ok) throw new Error("Failed to fetch books");
    
    const data = await res.json();

    return {
      total: data.total,
      books: query ? data.results : data.paginated_results
    };

  } catch (error) {
    console.error(error);
    return { total: 0, books: [] };
  }
}

const getImageUrl = (coverPath: string) => {
    if (!coverPath) return 'placeholder.jpg';
    let cleanPath = coverPath.replace(/\\/g, '/');
    if (cleanPath.startsWith('http')) return cleanPath;
    return cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`;
};

export default async function BookCollectionPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string; q?: string }>; 
}) {
  
  const resolvedSearchParams = await searchParams;
  
  const currentPage = Number(resolvedSearchParams.page) || 1;
  const searchQuery = resolvedSearchParams.q || "";

  const { books: currentBooks, total } = await fetchBooks(currentPage, searchQuery);

  const ITEMS_PER_PAGE = 15; 
  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);

  return (
    <section className="w-screen relative left-[50%] right-[50%] -ml-[50vw] -mr-[50vw] py-8">
      
      <div className="w-full px-6 md:px-12"> 
        
        <div className="w-full px-6 md:px-12 mb-10">
            <h1 className="text-3xl font-bold text-center">
                {searchQuery ? `Search Results: "${searchQuery}"` : "Book Collection"} 
                <span className="text-default-500 ml-2 text-xl">({total})</span>
            </h1>
        </div>

        {currentBooks.length === 0 ? (
            <div className="text-center py-20 text-default-500">
                <p>Tidak ada buku yang ditemukan.</p>
            </div>
        ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-6 w-full px-6 md:px-12">
            {currentBooks.map((item) => {
                const imageUrl = getImageUrl(item.cover);

                return (
                <Card
                    key={item.id}
                    isBlurred
                    className="border-none bg-background/60 dark:bg-default-100/50 w-full hover:bg-content2/50 transition-all duration-300"
                    shadow="sm"
                >
                    <CardBody className="p-0 overflow-hidden">
                    <div className="flex flex-row h-full">
                        
                        {/* Gambar */}
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

                        {/* Konten */}
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
                            </div>
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
        )}

        {totalPages > 1 && (
          <div className="mt-12 flex justify-center w-full">
            <BottomPagination totalPages={totalPages} />
          </div>
        )}
      </div>
    </section>
  );
}