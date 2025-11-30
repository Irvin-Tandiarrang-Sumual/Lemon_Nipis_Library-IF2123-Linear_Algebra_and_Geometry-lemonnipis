"use client";

import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Image } from "@heroui/image";
import Link from "next/link";
import { Chip } from "@heroui/chip";

const inputImage = {
  title: "Gambar yang Anda Upload",
  desc: "Mencari buku berdasarkan cover ini...",
  img: "https://nextui.org/images/hero-card-complete.jpeg", // dummy nanti ganti dgn path gambar yg diupload
};

// dummy later change w fetch first
const searchResults = [
  { id: "1", title: "The Design of Everyday Things", cover: "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1442460745i/840.jpg", txt: "Buku tentang desain produk yang sangat populer..." },
  { id: "2", title: "Don't Make Me Think", cover: "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1384288673i/18197267.jpg", txt: "Panduan usability web yang legendaris..." },
  { id: "3", title: "Sprint", cover: "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1457284924i/27785494.jpg", txt: "Cara memecahkan masalah besar dan menguji ide baru hanya dalam lima hari..." },
  { id: "4", title: "Hooked", cover: "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1403130833i/22611579.jpg", txt: "Bagaimana membangun produk yang membentuk kebiasaan..." },
  { id: "5", title: "Clean Code", cover: "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1436202607i/3735293.jpg", txt: "Panduan menulis kode yang rapi dan mudah dipelihara..." },
];

export default function SearchResultPage() {
  return (
    <section className="w-full py-8 min-h-screen">
      <div className="w-full px-6 md:px-12 max-w-[1600px] mx-auto">
        
        <div className="mb-8">
            <h1 className="text-3xl font-bold">Hasil Pencarian</h1>
            <p className="text-default-500 mt-2">Menampilkan buku yang mirip dengan gambar yang Anda unggah.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          <div className="lg:col-span-3">
             <div className="sticky top-24">
                <div className="mb-4 flex items-center gap-2">
                    <Chip color="primary" variant="dot" className="font-bold">Target Pencarian</Chip>
                </div>
                
                <Card className="border-none bg-background/60 dark:bg-default-100/50 w-full shadow-md">
                    <CardBody className="p-4">
                        <div className="bg-black/5 dark:bg-white/5 rounded-xl p-4 flex items-center justify-center aspect-[3/4] mb-4">
                            <Image
                                alt="Uploaded Image"
                                className="w-full h-full object-contain"
                                src={inputImage.img}
                                width="100%"
                                height="100%"
                                removeWrapper
                            />
                        </div>
                        <h3 className="text-lg font-bold text-center">{inputImage.title}</h3>
                        <p className="text-sm text-default-500 text-center mt-1">{inputImage.desc}</p>
                    </CardBody>
                </Card>

             </div>
          </div>

          <div className="lg:col-span-9">
             <div className="mb-4 flex items-center justify-between">
                <h2 className="text-xl font-bold">Ditemukan {searchResults.length} Buku Serupa</h2>
             </div>

             <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {searchResults.map((item) => (
                    <Card
                        key={item.id}
                        isBlurred
                        className="border-none bg-background/60 dark:bg-default-100/50 w-full hover:bg-content2/50 transition-all duration-300"
                        shadow="sm"
                    >
                        <CardBody className="p-0 overflow-hidden">
                            <div className="flex flex-row h-full min-h-[200px]">
                                
                                <div className="w-32 sm:w-40 relative flex-shrink-0 bg-black/5 dark:bg-white/5 p-2 flex items-center justify-center">
                                    <Image
                                        alt={item.title}
                                        className="object-contain w-full h-full rounded-md"
                                        src={item.cover}
                                        radius="none"
                                        width="100%"
                                        height="100%"
                                        removeWrapper
                                    />
                                </div>

                                <div className="flex flex-col justify-between p-4 flex-grow">
                                    <div>
                                        <div className="flex justify-between items-start w-full">
                                            <div className="flex flex-col pr-2 w-full">
                                                <span className="text-[10px] uppercase font-bold tracking-widest text-default-500 mb-1">
                                                    Kecocokan 98% • #{item.id}
                                                </span>
                                                <Link href={`/book-collection/${item.id}`} className="group block">
                                                    <h3 className="text-lg sm:text-xl font-bold text-foreground leading-tight line-clamp-2 group-hover:text-primary transition-colors" title={item.title}>
                                                        {item.title}
                                                    </h3>
                                                </Link>
                                            </div>
                                        </div>
                                        
                                        <p className="text-small text-default-500 line-clamp-3 mt-2 hidden sm:block">
                                            {item.txt}
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
                                            Lihat Buku
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </CardBody>
                    </Card>
                ))}
             </div>
          </div>

        </div>
      </div>
    </section>
  );
}