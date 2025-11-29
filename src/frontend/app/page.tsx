import { CustomSearchInput } from "@/components/search-input";
import { title, subtitle } from "@/components/primitives";

export default function Home() {
  return (
    <section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10">
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
