"use client";

import { Pagination } from "@heroui/pagination";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

export const BottomPagination = ({ totalPages }: { totalPages: number }) => {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const currentPage = Number(searchParams.get("page")) || 1;

  const handlePageChange = (page: number) => {
    const params = new URLSearchParams(searchParams);
    params.set("page", page.toString());

    router.replace(`${pathname}?${params.toString()}`);
  };

  return (
    <div className="flex justify-center my-8">
      <Pagination
        showControls
        color="primary"
        page={currentPage}      
        total={totalPages}    
        onChange={handlePageChange} 
      />
    </div>
  );
};