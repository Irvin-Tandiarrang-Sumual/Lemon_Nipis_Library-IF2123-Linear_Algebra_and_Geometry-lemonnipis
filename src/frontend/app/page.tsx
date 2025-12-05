import { CustomSearchInput } from "@/components/search-input";
import { title, subtitle } from "@/components/primitives";
import Image from "next/image";
export default function Home() {
  return (
    <section className="flex flex-col items-center justify-center gap-4 py-4 md:py-5">
      <div className="mb-2">
        <Image
          src="/LemonNipis.png"
          alt="Lemon Nipis Logo"
          width={350}
          height={350}
          priority
          className="drop-shadow-lg"
        />
      </div>
      <div className="inline-block max-w-xl text-center justify-center">
        <span className={title()}>Welcome to&nbsp;</span>
        <span className={title({ color: "lemon" })}>LemonNipis&nbsp;</span>
        <span className={title()}>Library&nbsp;</span>
        <div className={subtitle({ class: "mt-4" })}>
          "Jika kamu tidak suka membaca, kamu belum menemukan buku yang tepat." - J. K. Rowling, penulis buku Harry Potter
        </div>
      </div>

      <div className="w-full max-w-2xl px-6">
        <CustomSearchInput className="shadow-lg" />
      </div>
    </section>
  );
}